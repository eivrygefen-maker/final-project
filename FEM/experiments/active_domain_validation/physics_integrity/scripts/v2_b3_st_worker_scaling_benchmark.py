#!/usr/bin/env python3
"""Dev-only ST multi-target worker scaling benchmark (process-level shards; no FEM rebuild in workers)."""
from __future__ import annotations

import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import run_v2_B3_trace_coupled_operator_and_seed_transfer_audit as audit
import v2_b3_dev_solver_benchmark as dev_bench
import v2_b3_lmid_overnight_validation as lmid_overnight
from v2_b3_operator_build_profiler import B3OperatorBuildProfiler

B3_ST_WORKER_SCALING_ARG = "--B3-Lmid-ST-multi-target-worker-scaling-benchmark-only"
B3_ST_WORKER_COUNT_ARG = "--B3-ST-worker-count"
B3_ST_TARGETS_HZ_ARG = "--B3-ST-targets-hz"
B3_ST_SCALING_MESH_ARG = "--B3-ST-scaling-mesh-level"
B3_ST_WORKER_SHARD_EXECUTE_ARG = "--B3-ST-worker-shard-execute-only"
B3_ST_WORKER_SHARD_JOB_ARG = "--B3-ST-worker-shard-job-json"
B3_ST_REUSE_CHECKPOINT_ARG = "--B3-ST-reuse-checkpoint-dir"

ALLOWED_MESH_LEVELS = frozenset({"L_mid", "L_dev_dense", "L_prod"})
DEFAULT_MESH_LEVEL = "L_mid"
LMID_FULL_TARGETS = list(lmid_overnight.LMID_ST_TARGETS_HZ)
DENSE_DEFAULT_TARGETS = list(lmid_overnight.DENSE_ST_TARGETS_HZ)
L_PROD_DEFAULT_TARGETS = list(LMID_FULL_TARGETS)

_AUDIT_SCRIPT = Path(__file__).resolve().parent / "run_v2_B3_trace_coupled_operator_and_seed_transfer_audit.py"

# Strip from worker subprocess env so plain-python children do not inherit OpenMPI/PMIx RTE state
# from a parent launched via mpiexec (causes ompi_rte_init / local rank failures).
_SANITIZE_ENV_PREFIXES = (
    "OMPI_",
    "PMI_",
    "PMIX_",
    "OPAL_",
    "ORTE_",
    "MPI_",
    "SLURM_",
    "HYDRA_",
    "I_MPI_",
    "UCX_",
)
_SANITIZE_ENV_EXACT = frozenset(
    {
        "PMIX_SERVER_URI",
        "PMIX_NAMESPACE",
        "PMIX_RANK",
    }
)


def is_st_worker_scaling_mode(argv: Sequence[str]) -> bool:
    return B3_ST_WORKER_SCALING_ARG in argv


def is_st_worker_shard_execute_mode(argv: Sequence[str]) -> bool:
    return B3_ST_WORKER_SHARD_EXECUTE_ARG in argv


def _detect_openmpi_or_pmix_launch_env() -> List[str]:
    """Environment keys suggesting the process was started under mpiexec/Slurm PMI."""
    found: List[str] = []
    for key in os.environ:
        if key in _SANITIZE_ENV_EXACT or any(key.startswith(p) for p in _SANITIZE_ENV_PREFIXES):
            found.append(key)
    return sorted(found)


def _sanitized_subprocess_env() -> Dict[str, str]:
    """Copy of os.environ with OpenMPI/PMIx/Slurm launcher keys removed for worker children."""
    removed: List[str] = []
    clean: Dict[str, str] = {}
    for key, value in os.environ.items():
        if key in _SANITIZE_ENV_EXACT or any(key.startswith(p) for p in _SANITIZE_ENV_PREFIXES):
            removed.append(key)
            continue
        clean[key] = value
    clean["B3_ST_WORKER_SUBPROCESS_SANITIZED_ENV"] = "1"
    return clean


def st_worker_scaling_mpi_world_ok() -> Tuple[bool, int]:
    """Dev benchmark allows plain python (serial MPI) or mpiexec -n 1; rejects size > 1."""
    try:
        size = int(MPI.COMM_WORLD.size)
    except Exception:
        size = -1
    return size == 1, size


def _parse_arg_value(argv: Sequence[str], flag: str) -> Optional[str]:
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
    return None


def _parse_hz_list(text: str) -> List[float]:
    out: List[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("empty target frequency list")
    return out


def _parse_targets_hz(argv: Sequence[str], *, mesh_level: str) -> List[float]:
    raw = _parse_arg_value(argv, B3_ST_TARGETS_HZ_ARG)
    if raw:
        return _parse_hz_list(raw)
    if mesh_level == "L_dev_dense":
        return list(DENSE_DEFAULT_TARGETS)
    return list(LMID_FULL_TARGETS)


def _parse_worker_count(argv: Sequence[str]) -> int:
    raw = _parse_arg_value(argv, B3_ST_WORKER_COUNT_ARG)
    if raw is None:
        raise ValueError(f"missing required {B3_ST_WORKER_COUNT_ARG}")
    wc = int(raw)
    if wc not in (1, 2, 3):
        raise ValueError(f"{B3_ST_WORKER_COUNT_ARG} must be 1, 2, or 3")
    return wc


def _parse_mesh_level(argv: Sequence[str]) -> str:
    raw = _parse_arg_value(argv, B3_ST_SCALING_MESH_ARG)
    level = str(raw or DEFAULT_MESH_LEVEL).strip()
    if level not in ALLOWED_MESH_LEVELS:
        raise ValueError(f"{B3_ST_SCALING_MESH_ARG} must be one of {sorted(ALLOWED_MESH_LEVELS)}")
    return level


def _parse_reuse_checkpoint_dir(argv: Sequence[str]) -> Optional[Path]:
    raw = _parse_arg_value(argv, B3_ST_REUSE_CHECKPOINT_ARG)
    if not raw:
        return None
    return Path(raw).resolve()


def _struct_active_count_policy(mesh_level: str) -> str:
    if mesh_level == "L_mid":
        return "L_mid_exact"
    return "mesh_independent"


def _scaling_mesh_path(mesh_level: str) -> Path:
    return audit.mesh_path(mesh_level, audit.CASE_ID)


def _st_scaling_operator_contract_pass(
    payload: Dict[str, Any], *, built: Dict[str, Any], mesh_level: str
) -> bool:
    """Operator contract for ST worker-scaling (mesh_independent for dev/L_prod; L_mid exact)."""
    dev_bench._dev_record_operator_contract(payload, built=built)
    active_dim = int(np.asarray(built["active_local"]).size)
    payload["B3_ST_scaling_active_dimension"] = active_dim
    payload["B3_ST_scaling_A_shape"] = audit._mat_shape(built["A_active"])
    payload["B3_ST_scaling_M_shape"] = audit._mat_shape(built["M_active"])
    payload["B3_ST_scaling_mesh_level"] = mesh_level
    payload["B3_ST_scaling_mesh_path"] = str(_scaling_mesh_path(mesh_level).resolve())

    if mesh_level == "L_mid":
        lmid_payload: Dict[str, Any] = {}
        ok = bool(lmid_overnight._lmid_operator_contract_pass(lmid_payload, built=built))
        payload.update(lmid_payload)
        return ok

    dim_fail = dev_bench._dev_record_active_dimension_target_range(payload, mesh_level)
    ok = bool(
        payload.get("B3_DEV_operator_contract_pass")
        and payload.get("B3_DEV_zero_row_column_cleanup_contract_pass")
        and not dim_fail
    )
    if mesh_level == "L_prod":
        payload["B3_ST_scaling_L_prod_operator_contract_pass"] = ok
        payload["B3_ST_scaling_L_prod_active_dimension"] = active_dim
    return ok


def _targets_stem_suffix(targets_hz: List[float]) -> str:
    parts = [f"{float(f):g}".replace(".", "p") for f in targets_hz]
    return "_hz_" + "_".join(parts)


def _out_json_benchmark(mesh_level: str, worker_count: int, targets_hz: List[float]) -> Path:
    stem = (
        f"v2_B3_{mesh_level}_ST_multi_target_worker_scaling_W{worker_count}"
        f"{_targets_stem_suffix(targets_hz)}"
    )
    return audit.CONV_DIAG / f"{stem}.json"


def _sequential_reference_json(mesh_level: str, targets_hz: List[float]) -> Path:
    if mesh_level == "L_mid":
        return lmid_overnight.OUT_JSON_LMID_ST
    return dev_bench._dev_out_json_st_multi_target(mesh_level, targets_hz)


def _checkpoint_dir(mesh_level: str, run_id: str) -> Path:
    d = audit.CONV_DIAG / f"st_worker_scaling_{mesh_level}_{run_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _peak_rss_mb() -> Optional[float]:
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return dev_bench._safe_float(float(ru.ru_maxrss) / 1024.0)
    except Exception:
        return None


_CHECKPOINT_METADATA_REQUIRED_KEYS = (
    "active_local",
    "inactive_local",
    "free_rows",
    "bc_rows",
    "u_idx",
    "p_idx",
    "n_w",
    "n_u_b3",
    "active_dimension",
    "A_shape",
    "M_shape",
)

_CHECKPOINT_METADATA_OPTIONAL_CAND_KEYS = (
    "inactive_structural_count",
    "inactive_pressure_count",
    "inactive_aup_overlap_count",
    "aup_supported_count",
    "parent_raw_Auu_exact_zero_count",
    "parent_raw_Auu_nonzero_count",
)


def _built_metadata_from_built(built: Dict[str, Any], *, mesh_level: str) -> Dict[str, Any]:
    def _arr(key: str) -> List[int]:
        return np.asarray(built[key], dtype=np.int32).ravel().tolist()

    meta: Dict[str, Any] = {
        "mesh_level": mesh_level,
        "struct_active_count_policy": _struct_active_count_policy(mesh_level),
        "active_dimension": int(np.asarray(built["active_local"]).size),
        "n_w": int(built["n_w"]),
        "n_u_b3": int(built["n_u_b3"]),
        "active_local": _arr("active_local"),
        "inactive_local": _arr("inactive_local"),
        "free_rows": _arr("free_rows"),
        "bc_rows": _arr("bc_rows"),
        "u_idx": _arr("u_idx"),
        "p_idx": _arr("p_idx"),
        "A_shape": audit._mat_shape(built["A_active"]),
        "M_shape": audit._mat_shape(built["M_active"]),
    }
    cand = built.get("cand") or {}
    if cand:
        meta["inactive_structural_count"] = int(cand.get("inactive_structural_count", 0))
        meta["inactive_pressure_count"] = int(cand.get("inactive_pressure_count", 0))
        meta["inactive_aup_overlap_count"] = int(cand.get("inactive_aup_overlap_count", 0))
        meta["aup_supported_count"] = int(cand.get("aup_supported_count", 0))
        meta["parent_raw_Auu_exact_zero_count"] = int(cand.get("parent_raw_Auu_exact_zero_count", 0))
        meta["parent_raw_Auu_nonzero_count"] = int(cand.get("parent_raw_Auu_nonzero_count", 0))
    else:
        inactive_n = int(np.asarray(built["inactive_local"], dtype=np.int32).size)
        meta["inactive_structural_count"] = inactive_n
        meta["inactive_pressure_count"] = 0
        meta["inactive_aup_overlap_count"] = 0
        meta["parent_raw_Auu_exact_zero_count"] = inactive_n
        meta["parent_raw_Auu_nonzero_count"] = 0
    return meta


def _normalize_checkpoint_metadata(meta: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], bool]:
    """Tolerate legacy checkpoints; derive cand counters from inactive_local when absent."""
    missing_required = [k for k in _CHECKPOINT_METADATA_REQUIRED_KEYS if k not in meta]
    if missing_required:
        return dict(meta), missing_required, False

    normalized: Dict[str, Any] = dict(meta)
    missing_optional: List[str] = []
    inactive_local = np.asarray(meta["inactive_local"], dtype=np.int32).ravel()
    inactive_n = int(inactive_local.size)

    if "inactive_structural_count" not in normalized:
        missing_optional.append("inactive_structural_count")
        normalized["inactive_structural_count"] = inactive_n
    if "inactive_pressure_count" not in normalized:
        missing_optional.append("inactive_pressure_count")
        normalized["inactive_pressure_count"] = 0
    if "inactive_aup_overlap_count" not in normalized:
        missing_optional.append("inactive_aup_overlap_count")
        normalized["inactive_aup_overlap_count"] = 0
    if "aup_supported_count" not in normalized:
        missing_optional.append("aup_supported_count")
        normalized["aup_supported_count"] = 0
    if "parent_raw_Auu_exact_zero_count" not in normalized:
        missing_optional.append("parent_raw_Auu_exact_zero_count")
        normalized["parent_raw_Auu_exact_zero_count"] = int(normalized["inactive_structural_count"])
    if "parent_raw_Auu_nonzero_count" not in normalized:
        missing_optional.append("parent_raw_Auu_nonzero_count")
        normalized["parent_raw_Auu_nonzero_count"] = 0

    active_dim = int(normalized.get("active_dimension", 0))
    active_local = np.asarray(normalized["active_local"], dtype=np.int32).ravel()
    schema_pass = bool(
        active_dim > 0
        and int(active_local.size) == active_dim
        and inactive_n >= 0
    )
    return normalized, missing_optional, schema_pass


def _cand_from_normalized_metadata(meta: Dict[str, Any], inactive_local: np.ndarray) -> Dict[str, Any]:
    inactive_struct = int(meta.get("inactive_structural_count", int(inactive_local.size)))
    return {
        "inactive_local": inactive_local,
        "inactive_structural_count": inactive_struct,
        "inactive_pressure_count": int(meta.get("inactive_pressure_count", 0)),
        "inactive_aup_overlap_count": int(meta.get("inactive_aup_overlap_count", 0)),
        "aup_supported_count": int(meta.get("aup_supported_count", 0)),
        "parent_raw_Auu_exact_zero_count": int(
            meta.get("parent_raw_Auu_exact_zero_count", inactive_struct)
        ),
        "parent_raw_Auu_nonzero_count": int(meta.get("parent_raw_Auu_nonzero_count", 0)),
    }


def _built_from_metadata(
    meta: Dict[str, Any], *, A_active: Any, M_active: Any
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    normalized, missing_optional, schema_pass = _normalize_checkpoint_metadata(meta)
    inactive_local = np.asarray(normalized["inactive_local"], dtype=np.int32).ravel()
    cand = _cand_from_normalized_metadata(normalized, inactive_local)
    built = {
        "A_active": A_active,
        "M_active": M_active,
        "active_local": np.asarray(normalized["active_local"], dtype=np.int32).ravel(),
        "inactive_local": inactive_local,
        "free_rows": np.asarray(normalized["free_rows"], dtype=np.int32).ravel(),
        "bc_rows": np.asarray(normalized["bc_rows"], dtype=np.int32).ravel(),
        "u_idx": np.asarray(normalized["u_idx"], dtype=np.int32).ravel(),
        "p_idx": np.asarray(normalized["p_idx"], dtype=np.int32).ravel(),
        "n_w": int(normalized["n_w"]),
        "n_u_b3": int(normalized["n_u_b3"]),
        "cand": cand,
    }
    reuse_diag = {
        "B3_ST_scaling_reuse_checkpoint_metadata_schema_pass": bool(schema_pass),
        "B3_ST_scaling_reuse_checkpoint_missing_optional_fields": list(missing_optional),
        "B3_ST_scaling_reuse_checkpoint_derived_inactive_structural_count": int(
            cand["inactive_structural_count"]
        ),
        "B3_ST_scaling_reuse_checkpoint_failure_reason": None
        if schema_pass
        else "checkpoint_metadata_schema_invalid",
    }
    return built, reuse_diag


def _operator_build_profile_json(mesh_level: str) -> Path:
    return audit.CONV_DIAG / f"v2_B3_{mesh_level}_operator_build_profile.json"


def _finalize_operator_build_profile(
    prof: Any,
    *,
    payload: Dict[str, Any],
    mesh_level: str,
) -> None:
    if prof is None or not getattr(prof, "enabled", False):
        return
    prof.export_to_payload()
    prof.print_table(mesh_level=mesh_level)
    profile_json = _operator_build_profile_json(mesh_level)
    prof.write_json(profile_json)
    payload["B3_PROFILE_operator_build_json"] = str(profile_json.resolve())


def _export_operators(checkpoint: Path, *, built: Dict[str, Any], mesh_level: str) -> Dict[str, Any]:
    a_path = checkpoint / "A_active.petsc.bin"
    m_path = checkpoint / "M_active.petsc.bin"
    meta_path = checkpoint / "built_metadata.json"
    A_active = built["A_active"]
    M_active = built["M_active"]
    for mat, path in ((A_active, a_path), (M_active, m_path)):
        viewer = PETSc.Viewer().createBinary(str(path), "w", comm=PETSc.COMM_WORLD)
        try:
            mat.view(viewer)
        finally:
            viewer.destroy()
    meta = _built_metadata_from_built(built, mesh_level=mesh_level)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "checkpoint_dir": str(checkpoint.resolve()),
        "A_active_binary": str(a_path.resolve()),
        "M_active_binary": str(m_path.resolve()),
        "built_metadata_json": str(meta_path.resolve()),
        **meta,
    }


def _verify_operator_export(checkpoint: Path) -> Tuple[bool, List[str], Dict[str, Any]]:
    """True when PETSc binary operator export artifacts are present on disk."""
    required = [
        checkpoint / "A_active.petsc.bin",
        checkpoint / "M_active.petsc.bin",
        checkpoint / "built_metadata.json",
    ]
    missing = [p.name for p in required if not p.is_file()]
    info_present = [
        (checkpoint / "A_active.petsc.bin.info").is_file(),
        (checkpoint / "M_active.petsc.bin.info").is_file(),
    ]
    export_pass = len(missing) == 0
    detail = {
        "checkpoint_dir": str(checkpoint.resolve()),
        "required_files": [p.name for p in required],
        "missing_files": missing,
        "petsc_info_sidecars_present": all(info_present),
    }
    return export_pass, missing, detail


def _load_operators(checkpoint: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    meta_path = checkpoint / "built_metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing built metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    a_path = checkpoint / "A_active.petsc.bin"
    m_path = checkpoint / "M_active.petsc.bin"
    if not a_path.is_file() or not m_path.is_file():
        raise FileNotFoundError(f"missing operator binary under {checkpoint}")

    def _load_mat(path: Path) -> Any:
        viewer = PETSc.Viewer().createBinary(str(path), "r", comm=PETSc.COMM_WORLD)
        try:
            mat = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
            mat.load(viewer)
            audit._petsc_mat_try_assemble(mat)
            return mat
        finally:
            viewer.destroy()

    A_active = _load_mat(a_path)
    M_active = _load_mat(m_path)
    built, reuse_diag = _built_from_metadata(meta, A_active=A_active, M_active=M_active)
    return built, meta, reuse_diag


def _partition_target_shards(
    targets_hz: List[float],
    worker_count: int,
) -> List[List[Tuple[int, float]]]:
    shards: List[List[Tuple[int, float]]] = [[] for _ in range(worker_count)]
    for ti, hz in enumerate(targets_hz):
        shards[ti % worker_count].append((ti, float(hz)))
    return shards


def _st_mumps_policy_spec(policy: str) -> Dict[str, Any]:
    """Solver-only MUMPS/PETSc option sets (ST prefix); does not alter FEM operators."""
    base = {
        "st_mat_mumps_icntl_6": 7,
        "st_mat_mumps_icntl_12": 1,
        "st_mat_mumps_icntl_7": 0,
        "st_mat_mumps_icntl_4": 0,
    }
    if policy == "default":
        return {
            **base,
            "st_mat_mumps_icntl_14": 500,
            "st_mat_mumps_icntl_24": 0,
        }
    if policy == "L_prod_relaxed":
        return {
            **base,
            "st_mat_mumps_icntl_14": 800,
            "st_mat_mumps_icntl_22": 1,
            "st_mat_mumps_icntl_23": 8192,
            "st_mat_mumps_icntl_24": 1,
            "st_mat_mumps_icntl_7": 7,
        }
    if policy == "L_prod_maximum":
        return {
            **base,
            "st_mat_mumps_icntl_14": 1200,
            "st_mat_mumps_icntl_22": 1,
            "st_mat_mumps_icntl_23": 16384,
            "st_mat_mumps_icntl_24": 1,
            "st_mat_mumps_icntl_7": 7,
            "st_mat_mumps_icntl_3": 0,
        }
    raise ValueError(f"unknown_mumps_policy={policy}")


def _apply_st_mumps_petsc_policy(policy: str) -> Dict[str, Any]:
    spec = _st_mumps_policy_spec(policy)
    petsc_opts = PETSc.Options()
    for key, val in spec.items():
        petsc_opts[key] = val
    return {"mumps_policy_applied": policy, "petsc_options_written": dict(spec)}


def _mumps_policy_chain(mesh_level: str) -> List[str]:
    if mesh_level == "L_prod":
        return ["default", "L_prod_relaxed", "L_prod_maximum"]
    return ["default"]


def _extract_st_failure_diagnostics(exc: BaseException) -> Dict[str, Any]:
    """Classify PETSc/SLEPc/MUMPS setup failures for per-target reporting."""
    msg = str(exc)
    lower = msg.lower()
    ierr = getattr(exc, "ierr", None)
    out: Dict[str, Any] = {
        "exception_type": type(exc).__name__,
        "exception_message": msg[:4096],
        "petsc_error_code": int(ierr) if ierr is not None else None,
        "mumps_infog1": None,
        "mumps_info2": None,
        "failure_class": "ST_SETUP_OR_SOLVE_UNKNOWN",
        "recommended_next_action": "inspect_worker_log_and_per_target_diagnostics",
    }
    if "infog(1)=-13" in lower or "infog[1]=-13" in lower:
        out["mumps_infog1"] = -13
        out["failure_class"] = "MUMPS_NUMERICAL_FACTORIZATION_OR_MEMORY"
        out["recommended_next_action"] = (
            "retry_with_L_prod_relaxed_or_maximum_mumps_policy;"
            "ensure_sufficient_RAM_or_out_of_core;"
            "consider_single_target_diagnostic_first"
        )
    if "error code 76" in lower or (ierr is not None and int(ierr) == 76):
        out["failure_class"] = "PETSC_PC_FACTOR_SETUp_FAILED"
        out["recommended_next_action"] = (
            "MUMPS_LU_factorization_failed_at_eps_setUp;"
            "try_relaxed_mumps_icntl_14_22_23"
        )
    if "sinvert" in lower and "setUp" in msg:
        out.setdefault("failure_class", "ST_SINVERT_SETUP_FAILED")
    if "memory" in lower or "alloc" in lower:
        out["failure_class"] = "MEMORY_ALLOCATION_OR_MUMPS_WORKSPACE"
        out["recommended_next_action"] = "reduce_concurrent_workers;use_mumps_out_of_core;increase_node_RAM"
    return out


def _configure_eps_st_sinvert(
    eps: Any,
    A_active: Any,
    M_active: Any,
    *,
    target_hz: float,
    target_lambda: float,
    mumps_policy: str,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    dev_bench._dev_configure_coarse_krylovschur_sinvert_eps(
        eps,
        A_active,
        M_active,
        payload=cfg,
        target_lambda=float(target_lambda),
        target_hz=float(target_hz),
    )
    mumps_written = _apply_st_mumps_petsc_policy(mumps_policy)
    cfg.update(mumps_written)
    return cfg


def _run_st_targets_on_built(
    built: Dict[str, Any],
    target_entries: List[Tuple[int, float]],
    *,
    mesh_level: str = "L_mid",
    payload_prefix: str = "B3_ST_scaling",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float, float, Dict[str, Any]]:
    """Run ST multi-target slice; continue on per-target failure; optional L_prod MUMPS fallback."""
    from slepc4py import SLEPc

    freq_lo = float(audit.B3_CISS_VALIDATION_FREQ_LO_HZ)
    freq_hi = float(audit.B3_CISS_VALIDATION_FREQ_HI_HZ)
    A_active = built["A_active"]
    M_active = built["M_active"]
    per_target: Dict[str, Any] = {}
    all_accepted: List[Dict[str, Any]] = []
    total_setup_s = 0.0
    total_solve_s = 0.0
    policy_chain = _mumps_policy_chain(mesh_level)
    summary: Dict[str, Any] = {
        "B3_ST_scaling_solve_loop_entered_pass": True,
        "B3_ST_scaling_targets_attempted_count": 0,
        "B3_ST_scaling_targets_setup_succeeded_count": 0,
        "B3_ST_scaling_targets_solve_attempted_count": 0,
        "B3_ST_scaling_targets_solved_count": 0,
        "B3_ST_scaling_mumps_policy_chain": policy_chain,
        "B3_ST_scaling_skip_reason": None,
    }

    for ti, target_hz in target_entries:
        key = f"{payload_prefix}_target_{ti}_"
        target_lambda = float(audit._b3_hz_to_lambda_sq(float(target_hz)))
        per_target[f"{key}target_frequency_hz"] = float(target_hz)
        per_target[f"{key}target_lambda"] = dev_bench._safe_float(target_lambda)
        per_target[f"{key}setup_attempted"] = True
        per_target[f"{key}setup_succeeded"] = False
        per_target[f"{key}solve_attempted"] = False
        per_target[f"{key}solve_succeeded"] = False
        summary["B3_ST_scaling_targets_attempted_count"] = int(summary["B3_ST_scaling_targets_attempted_count"]) + 1

        eps = None
        setup_succeeded = False
        last_setup_exc: Optional[BaseException] = None
        policies_tried: List[str] = []
        setup_meta: Dict[str, Any] = {}

        for policy in policy_chain:
            policies_tried.append(policy)
            if eps is not None:
                try:
                    eps.destroy()
                except Exception:
                    pass
                eps = None
            try:
                eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
                setup_meta = _configure_eps_st_sinvert(
                    eps,
                    A_active,
                    M_active,
                    target_hz=float(target_hz),
                    target_lambda=target_lambda,
                    mumps_policy=policy,
                )
                t0 = time.perf_counter()
                eps.setUp()
                setup_s = time.perf_counter() - t0
                setup_succeeded = True
                per_target[f"{key}setup_succeeded"] = True
                per_target[f"{key}setup_elapsed_seconds"] = dev_bench._safe_float(setup_s)
                per_target[f"{key}mumps_policy_effective"] = policy
                per_target[f"{key}mumps_petsc_options_effective"] = setup_meta.get("petsc_options_written")
                per_target[f"{key}mumps_policies_tried"] = list(policies_tried)
                intro = dev_bench._dev_introspect_st_targeting_after_setup(eps)
                per_target[f"{key}effective_target"] = intro.get("B3_DEV_ST_target_effective")
                per_target[f"{key}effective_shift"] = intro.get("B3_DEV_ST_shift_effective")
                per_target[f"{key}effective_which"] = intro.get("B3_DEV_ST_which_effective_normalized")
                total_setup_s += setup_s
                summary["B3_ST_scaling_targets_setup_succeeded_count"] = int(
                    summary["B3_ST_scaling_targets_setup_succeeded_count"]
                ) + 1
                break
            except Exception as exc:
                last_setup_exc = exc
                diag = _extract_st_failure_diagnostics(exc)
                per_target[f"{key}mumps_policies_tried"] = list(policies_tried)
                per_target[f"{key}last_failed_mumps_policy"] = policy
                per_target.update({f"{key}{k}": v for k, v in diag.items() if not k.startswith("exception_")})
                per_target[f"{key}failure_reason"] = f"{type(exc).__name__}:{exc}"

        if not setup_succeeded:
            if last_setup_exc is not None:
                diag = _extract_st_failure_diagnostics(last_setup_exc)
                per_target.update({f"{key}{k}": v for k, v in diag.items() if not k.startswith("exception_")})
            per_target[f"{key}solve_pass"] = False
            if eps is not None:
                try:
                    eps.destroy()
                except Exception:
                    pass
            continue

        per_target[f"{key}solve_attempted"] = True
        summary["B3_ST_scaling_targets_solve_attempted_count"] = int(
            summary["B3_ST_scaling_targets_solve_attempted_count"]
        ) + 1
        try:
            t1 = time.perf_counter()
            eps.solve()
            solve_s = time.perf_counter() - t1
            per_target[f"{key}solve_elapsed_seconds"] = dev_bench._safe_float(solve_s)
            per_target[f"{key}solve_succeeded"] = True
            per_target[f"{key}solve_pass"] = True
            total_solve_s += solve_s
            nconv, accepted = dev_bench._dev_collect_accepted_st_modes(
                eps,
                A_active,
                built,
                target_hz=float(target_hz),
                freq_lo=freq_lo,
                freq_hi=freq_hi,
            )
            per_target[f"{key}converged_mode_count"] = int(nconv)
            per_target[f"{key}accepted_mode_count_in_interval"] = int(len(accepted))
            per_target[f"{key}accepted_frequencies"] = [float(m["frequency_hz"]) for m in accepted]
            all_accepted.extend(accepted)
            summary["B3_ST_scaling_targets_solved_count"] = int(summary["B3_ST_scaling_targets_solved_count"]) + 1
        except Exception as exc:
            diag = _extract_st_failure_diagnostics(exc)
            per_target[f"{key}solve_pass"] = False
            per_target[f"{key}solve_succeeded"] = False
            per_target[f"{key}failure_reason"] = f"{type(exc).__name__}:{exc}"
            per_target.update({f"{key}{k}": v for k, v in diag.items() if not k.startswith("exception_")})
            per_target[f"{key}failure_class"] = diag.get("failure_class", "ST_SOLVE_FAILED")
        finally:
            if eps is not None:
                try:
                    eps.destroy()
                except Exception:
                    pass

    if int(summary["B3_ST_scaling_targets_attempted_count"]) == 0:
        summary["B3_ST_scaling_skip_reason"] = "empty_target_list"
    return all_accepted, per_target, total_setup_s, total_solve_s, summary


def _derive_loop_summary_from_per_target(
    per_target: Dict[str, Any],
    targets_hz: List[float],
    *,
    payload_prefix: str = "B3_ST_scaling",
) -> Dict[str, Any]:
    attempted = setup_ok = solve_attempted = solved = 0
    for ti in range(len(targets_hz)):
        key = f"{payload_prefix}_target_{ti}_"
        if per_target.get(f"{key}setup_attempted"):
            attempted += 1
        if per_target.get(f"{key}setup_succeeded"):
            setup_ok += 1
        if per_target.get(f"{key}solve_attempted"):
            solve_attempted += 1
        if per_target.get(f"{key}solve_succeeded"):
            solved += 1
    return {
        "B3_ST_scaling_solve_loop_entered_pass": attempted > 0 or len(targets_hz) > 0,
        "B3_ST_scaling_targets_attempted_count": attempted,
        "B3_ST_scaling_targets_setup_succeeded_count": setup_ok,
        "B3_ST_scaling_targets_solve_attempted_count": solve_attempted,
        "B3_ST_scaling_targets_solved_count": solved,
        "B3_ST_scaling_skip_reason": None if attempted > 0 else "no_per_target_setup_attempted",
    }


def _record_solve_loop_status(
    payload: Dict[str, Any],
    *,
    loop_summary: Dict[str, Any],
    targets_hz: List[float],
) -> None:
    payload.update(loop_summary)
    attempted = int(payload.get("B3_ST_scaling_targets_attempted_count") or 0)
    if attempted == 0:
        payload["B3_ST_scaling_solve_loop_entered_pass"] = bool(
            payload.get("B3_ST_scaling_solve_loop_entered_pass")
        )
        if not payload.get("B3_ST_scaling_skip_reason"):
            payload["B3_ST_scaling_skip_reason"] = "no_targets_in_run_plan"


def _finalize_scaling_verdict(
    payload: Dict[str, Any],
    *,
    targets_hz: List[float],
) -> Tuple[str, int]:
    """Verdict from solve outcomes; reference JSON only affects parity fields."""
    if not payload.get("B3_ST_scaling_solve_loop_entered_pass"):
        payload["B3_ST_scaling_skip_reason"] = payload.get("B3_ST_scaling_skip_reason") or "solve_loop_never_entered"
        return "B3_ST_WORKER_SCALING_SOLVE_LOOP_NEVER_ENTERED", 2

    attempted = int(payload.get("B3_ST_scaling_targets_attempted_count") or 0)
    if attempted == 0:
        payload["B3_ST_scaling_skip_reason"] = payload.get("B3_ST_scaling_skip_reason") or "no_targets_attempted"
        return "B3_ST_WORKER_SCALING_NO_TARGETS_ATTEMPTED", 2

    setup_ok = int(payload.get("B3_ST_scaling_targets_setup_succeeded_count") or 0)
    solved = int(payload.get("B3_ST_scaling_targets_solved_count") or 0)
    n_unique = int(payload.get("B3_ST_scaling_unique_accepted_frequency_count") or 0)
    mesh_level = str(payload.get("B3_ST_scaling_mesh_level") or "")

    if setup_ok == 0 and attempted > 0:
        if mesh_level == "L_prod":
            return "B3_ST_WORKER_SCALING_L_PROD_ALL_TARGETS_SETUP_FACTORIZATION_FAILED", 2
        return "B3_ST_WORKER_SCALING_ALL_TARGETS_SETUP_FACTORIZATION_FAILED", 2

    if solved == 0 and attempted > 0:
        return "B3_ST_WORKER_SCALING_ALL_TARGETS_SOLVE_FAILED", 2

    parity = payload.get("frequency_parity_pass")
    if parity is True:
        return "B3_ST_WORKER_SCALING_PASS", 0
    if parity is False:
        return "B3_ST_WORKER_SCALING_FREQUENCY_MISMATCH", 2
    if n_unique > 0:
        if payload.get("sequential_comparison_skipped_reason"):
            return "B3_ST_WORKER_SCALING_PARTIAL_PASS_COMPARISON_SKIPPED", 0
        if payload.get("sequential_reference_missing"):
            return "B3_ST_WORKER_SCALING_PARTIAL_PASS_NO_REFERENCE", 0
        return "B3_ST_WORKER_SCALING_PARTIAL_PASS_NO_REFERENCE", 0
    return "B3_ST_WORKER_SCALING_COMPLETED_ZERO_ACCEPTED_MODES", 2


def _aggregate_union_and_provenance(
    all_accepted: List[Dict[str, Any]],
) -> Tuple[List[float], List[Dict[str, Any]]]:
    union_freqs = dev_bench._dev_deduplicate_frequencies_hz(
        [float(m["frequency_hz"]) for m in all_accepted],
        tol_hz=dev_bench.B3_DEV_ST_MULTI_DEDUP_TOL_HZ,
    )
    provenance = lmid_overnight._st_deduplicated_provenance(
        all_accepted,
        union_freqs,
        tol_hz=dev_bench.B3_DEV_ST_MULTI_DEDUP_TOL_HZ,
    )
    return union_freqs, provenance


def _targets_lists_match(a: List[float], b: List[float], *, rtol: float = 1.0e-9) -> bool:
    if len(a) != len(b):
        return False
    return all(abs(float(x) - float(y)) <= rtol * max(1.0, abs(float(y))) for x, y in zip(a, b))


def _compare_to_sequential_reference(
    union_freqs: List[float],
    *,
    ref_path: Path,
    targets_hz: List[float],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "sequential_reference_json": str(ref_path),
        "sequential_reference_loaded": False,
        "benchmark_targets_hz": list(targets_hz),
    }
    if not ref_path.is_file():
        out["sequential_reference_missing"] = True
        out["frequency_parity_pass"] = None
        out["sequential_comparison_available"] = False
        return out
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    ref_targets = list(
        ref.get("B3_Lmid_ST_multi_target_targets_hz")
        or ref.get("B3_DEV_ST_multi_target_targets_hz")
        or []
    )
    if ref_targets and not _targets_lists_match(list(targets_hz), ref_targets):
        out["sequential_reference_loaded"] = True
        out["sequential_comparison_available"] = False
        out["sequential_comparison_skipped_reason"] = "benchmark_targets_differ_from_sequential_reference_targets"
        out["sequential_reference_targets_hz"] = ref_targets
        out["frequency_parity_pass"] = None
        return out
    ref_freqs = list(
        ref.get("B3_Lmid_ST_multi_target_unique_accepted_frequencies")
        or ref.get("B3_DEV_ST_multi_target_unique_accepted_frequencies")
        or []
    )
    if not ref_freqs:
        ref_freqs = lmid_overnight._accepted_frequencies_from_mode_payload(ref)
    out["sequential_reference_loaded"] = True
    out["sequential_comparison_available"] = True
    out["sequential_unique_accepted_frequency_count"] = len(ref_freqs)
    out["benchmark_unique_accepted_frequency_count"] = len(union_freqs)
    out["same_unique_frequency_count_pass"] = bool(len(ref_freqs) == len(union_freqs))
    matches, missing, extra = dev_bench._dev_compare_frequency_sets(
        union_freqs,
        ref_freqs,
        match_tol_hz=dev_bench.B3_DEV_ST_MULTI_CISS_MATCH_TOL_HZ,
    )
    out["matched_frequency_count_vs_sequential"] = int(matches)
    out["missing_frequencies_vs_sequential"] = missing
    out["extra_frequencies_vs_sequential"] = extra
    out["frequency_parity_pass"] = bool(
        len(missing) == 0 and len(extra) == 0 and len(union_freqs) == len(ref_freqs)
    )
    return out


def _worker_artifact_paths(checkpoint: Path, worker_id: int) -> Dict[str, Path]:
    return {
        "job": checkpoint / f"worker_{worker_id}_job.json",
        "shard": checkpoint / f"worker_{worker_id}_shard.json",
        "failure": checkpoint / f"worker_{worker_id}_failure.json",
        "log": checkpoint / f"worker_{worker_id}.log",
    }


def _write_worker_job_files(
    checkpoint: Path,
    *,
    shards: List[List[Tuple[int, float]]],
    worker_count: int,
    mesh_level: str,
) -> List[Dict[str, Any]]:
    """Write worker_{id}_job.json for every worker slot before any subprocess launch."""
    specs: List[Dict[str, Any]] = []
    for wi in range(worker_count):
        entries = shards[wi] if wi < len(shards) else []
        paths = _worker_artifact_paths(checkpoint, wi)
        job = {
            "worker_id": wi,
            "worker_count": worker_count,
            "mesh_level": mesh_level,
            "checkpoint_dir": str(checkpoint.resolve()),
            "target_entries": [[int(ti), float(hz)] for ti, hz in entries],
            "target_indices": [int(ti) for ti, _ in entries],
            "target_frequencies_hz": [float(hz) for _, hz in entries],
            "shard_nonempty": bool(entries),
            "output_shard_json": str(paths["shard"].resolve()),
            "output_failure_json": str(paths["failure"].resolve()),
            "log_path": str(paths["log"].resolve()),
        }
        paths["job"].write_text(json.dumps(job, indent=2), encoding="utf-8")
        specs.append({"worker_id": wi, "entries": entries, "paths": paths, "job": job})
    return specs


def _spawn_worker_shard_plain_python(job_path: Path, log_path: Path) -> int:
    """Launch worker as plain python with sanitized env (no mpiexec, no inherited OpenMPI RTE)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_AUDIT_SCRIPT),
        B3_ST_WORKER_SHARD_EXECUTE_ARG,
        B3_ST_WORKER_SHARD_JOB_ARG,
        str(job_path.resolve()),
    ]
    child_env = _sanitized_subprocess_env()
    removed_keys = sorted(
        k
        for k in os.environ
        if k in _SANITIZE_ENV_EXACT or any(k.startswith(p) for p in _SANITIZE_ENV_PREFIXES)
    )
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"# command: {' '.join(cmd)}\n")
        logf.write(f"# sanitized_env_removed_key_count: {len(removed_keys)}\n")
        if removed_keys:
            logf.write("# sanitized_env_removed_keys_sample:\n")
            for key in removed_keys[:64]:
                logf.write(f"#   {key}\n")
            if len(removed_keys) > 64:
                logf.write(f"#   ... and {len(removed_keys) - 64} more\n")
        logf.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(_AUDIT_SCRIPT.parent),
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            env=child_env,
        )
        logf.write(f"\n# exit_code: {proc.returncode}\n")
    return int(proc.returncode)


def _ensure_worker_failure_artifact(
    paths: Dict[str, Path],
    *,
    worker_id: int,
    worker_count: int,
    return_code: int,
    reason: str,
) -> None:
    if paths["shard"].is_file() or paths["failure"].is_file():
        return
    _write_worker_failure_json(
        paths["failure"],
        {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": "B3_ST_worker_shard_parent_synthesized_failure",
            "worker_id": worker_id,
            "worker_count": worker_count,
            "failure_reason": reason,
            "worker_return_code": int(return_code),
            "artifact": "worker_failure_json",
            "next_step_verdict": "B3_ST_WORKER_SHARD_FAILED",
        },
    )


def _collect_worker_process_status(
    checkpoint: Path,
    worker_specs: List[Dict[str, Any]],
    *,
    worker_return_codes: Optional[Dict[int, int]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate logs and shard/failure JSON for each worker that had targets."""
    missing_files: List[str] = []
    worker_details: List[Dict[str, Any]] = []
    all_pass = True
    failure_reasons: List[str] = []

    for spec in worker_specs:
        wi = int(spec["worker_id"])
        paths = spec["paths"]
        entries = spec["entries"]
        detail: Dict[str, Any] = {
            "worker_id": wi,
            "shard_nonempty": bool(entries),
            "job_json": paths["job"].name,
            "log_json": paths["log"].name,
        }
        if not paths["job"].is_file():
            missing_files.append(paths["job"].name)
            all_pass = False

        if not entries:
            detail["skipped"] = True
            detail["skip_reason"] = "empty_target_shard"
            worker_details.append(detail)
            continue

        if worker_return_codes is not None and wi in worker_return_codes:
            detail["subprocess_return_code"] = int(worker_return_codes[wi])
            if int(worker_return_codes[wi]) != 0:
                all_pass = False
                failure_reasons.append(f"worker_{wi}_subprocess_rc={worker_return_codes[wi]}")

        if not paths["log"].is_file():
            missing_files.append(paths["log"].name)
            all_pass = False

        has_shard = paths["shard"].is_file()
        has_failure = paths["failure"].is_file()
        detail["shard_json_present"] = has_shard
        detail["failure_json_present"] = has_failure

        if not has_shard and not has_failure:
            missing_files.append(paths["shard"].name)
            missing_files.append(paths["failure"].name)
            all_pass = False
            failure_reasons.append(f"worker_{wi}_missing_shard_and_failure_json")
        elif has_failure and not has_shard:
            try:
                fail_body = json.loads(paths["failure"].read_text(encoding="utf-8"))
                detail["failure_reason"] = fail_body.get("failure_reason")
                detail["worker_return_code"] = fail_body.get("worker_return_code")
            except Exception as exc:
                detail["failure_reason"] = f"failure_json_unreadable:{exc}"
            all_pass = False
            failure_reasons.append(f"worker_{wi}_failed")
        elif has_shard:
            detail["worker_pass"] = True

        worker_details.append(detail)

    status = {
        "worker_processes_pass": bool(all_pass),
        "worker_processes_failure_reason": "; ".join(failure_reasons) if failure_reasons else None,
        "worker_processes_missing_files": missing_files,
        "worker_process_details": worker_details,
    }
    return all_pass, status


def _write_worker_failure_json(path: Path, body: Dict[str, Any]) -> None:
    audit._write_json_atomic(path, body)


def run_st_worker_shard_execute(argv: Sequence[str]) -> int:
    if MPI.COMM_WORLD.size not in (1,):
        print(
            f"[B3_ST_scaling] worker shard requires single-rank MPI (size={MPI.COMM_WORLD.size}); "
            "use plain python worker launch",
            flush=True,
        )
        return 2
    job_raw = _parse_arg_value(argv, B3_ST_WORKER_SHARD_JOB_ARG)
    if not job_raw:
        print(f"[B3_ST_scaling] missing {B3_ST_WORKER_SHARD_JOB_ARG}", flush=True)
        return 2
    job_path = Path(job_raw)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    checkpoint = Path(str(job["checkpoint_dir"]))
    worker_id = int(job["worker_id"])
    worker_count = int(job["worker_count"])
    target_entries = [(int(ti), float(hz)) for ti, hz in job["target_entries"]]
    out_shard = Path(str(job["output_shard_json"]))
    out_failure = Path(str(job.get("output_failure_json") or checkpoint / f"worker_{worker_id}_failure.json"))

    mats: List[Any] = []
    shard_payload: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "B3_ST_worker_shard_execute_only",
        "worker_id": worker_id,
        "worker_count": worker_count,
        "checkpoint_dir": str(checkpoint.resolve()),
        "target_indices": [ti for ti, _ in target_entries],
        "target_frequencies_hz": [hz for _, hz in target_entries],
        "worker_launch": "plain_python_subprocess",
        "production_promotion": "BLOCKED",
    }
    rc = 2
    if not target_entries:
        shard_payload["failure_reason"] = "empty_target_shard"
        shard_payload["next_step_verdict"] = "B3_ST_WORKER_SHARD_SKIPPED_EMPTY"
        _write_worker_failure_json(
            out_failure,
            {
                **shard_payload,
                "worker_return_code": 2,
                "artifact": "worker_failure_json",
            },
        )
        return 2

    try:
        built, meta, reuse_diag = _load_operators(checkpoint)
        mats.extend([built["A_active"], built["M_active"]])
        shard_payload.update(reuse_diag)
        if not reuse_diag.get("B3_ST_scaling_reuse_checkpoint_metadata_schema_pass"):
            shard_payload["failure_reason"] = reuse_diag.get(
                "B3_ST_scaling_reuse_checkpoint_failure_reason"
            )
            _write_worker_failure_json(
                out_failure,
                {**shard_payload, "worker_return_code": 2, "artifact": "worker_failure_json"},
            )
            return 2
        shard_payload["loaded_active_dimension"] = int(meta.get("active_dimension", 0))
        mesh_level_job = str(job.get("mesh_level") or meta.get("mesh_level") or "L_mid")
        all_accepted, per_target, setup_s, solve_s, loop_summary = _run_st_targets_on_built(
            built,
            target_entries,
            mesh_level=mesh_level_job,
            payload_prefix="B3_ST_scaling",
        )
        shard_payload.update(per_target)
        shard_payload.update(loop_summary)
        union_freqs, provenance = _aggregate_union_and_provenance(all_accepted)
        shard_payload["B3_ST_scaling_shard_accepted_mode_records"] = all_accepted
        shard_payload["B3_ST_scaling_shard_unique_accepted_frequencies"] = union_freqs
        shard_payload["B3_ST_scaling_shard_unique_accepted_frequency_count"] = len(union_freqs)
        shard_payload["B3_ST_scaling_shard_deduplicated_mode_provenance"] = provenance
        shard_payload["B3_ST_scaling_shard_total_setup_elapsed_seconds"] = dev_bench._safe_float(setup_s)
        shard_payload["B3_ST_scaling_shard_total_solve_elapsed_seconds"] = dev_bench._safe_float(solve_s)
        shard_payload["B3_ST_scaling_shard_peak_rss_mb"] = _peak_rss_mb()
        shard_payload["next_step_verdict"] = "B3_ST_WORKER_SHARD_PASS"
        audit._write_json_atomic(out_shard, shard_payload)
        if out_failure.is_file():
            try:
                out_failure.unlink()
            except OSError:
                pass
        rc = 0
    except Exception as exc:
        fail_body = {
            **shard_payload,
            "failure_reason": f"{type(exc).__name__}:{exc}",
            "next_step_verdict": "B3_ST_WORKER_SHARD_FAILED",
            "worker_return_code": 2,
            "artifact": "worker_failure_json",
        }
        _write_worker_failure_json(out_failure, fail_body)
        if out_shard.is_file():
            try:
                out_shard.unlink()
            except OSError:
                pass
        rc = 2
    finally:
        for m in mats:
            try:
                m.destroy()
            except Exception:
                pass
    return rc


def run_st_worker_scaling_benchmark(argv: Sequence[str], pre: Dict[str, Any]) -> int:
    mpi_ok, mpi_size = st_worker_scaling_mpi_world_ok()
    if not pre.get("preassembly_contract_pass"):
        print("[B3_ST_scaling] preassembly_contract_pass=False", flush=True)
        return 2
    if not mpi_ok:
        print(
            f"[B3_ST_scaling] requires MPI COMM_WORLD size 1 (got {mpi_size}); "
            "use plain python or mpiexec -n 1",
            flush=True,
        )
        return 2
    try:
        mesh_level = _parse_mesh_level(argv)
        worker_count = _parse_worker_count(argv)
        targets_hz = _parse_targets_hz(argv, mesh_level=mesh_level)
    except ValueError as exc:
        print(f"[B3_ST_scaling] {exc}", flush=True)
        return 2

    inherited_launch_env = _detect_openmpi_or_pmix_launch_env()
    if worker_count > 1 and inherited_launch_env:
        print(
            "[B3_ST_scaling] WARN: parent has OpenMPI/PMIx launcher environment variables. "
            "Launch the benchmark with plain python (not mpiexec) so worker subprocesses "
            "do not inherit RTE state. Example:\n"
            "  python FEM/experiments/active_domain_validation/physics_integrity/scripts/"
            "run_v2_B3_trace_coupled_operator_and_seed_transfer_audit.py "
            f"{B3_ST_WORKER_SCALING_ARG} ...",
            flush=True,
        )

    if mesh_level in ("L_mid", "L_prod") and worker_count >= 3:
        print(
            f"[B3_ST_scaling] worker_count=3 is blocked on {mesh_level} (RAM risk). "
            "Use L_dev_dense for 3-worker smoke or worker_count<=2.",
            flush=True,
        )
        return 2

    mesh_file = _scaling_mesh_path(mesh_level)
    if not mesh_file.is_file():
        print(f"[B3_ST_scaling] mesh_missing={mesh_file}", flush=True)
        return 2

    reuse_ckpt = _parse_reuse_checkpoint_dir(argv)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    checkpoint = reuse_ckpt if reuse_ckpt is not None else _checkpoint_dir(mesh_level, run_id)
    out_json = _out_json_benchmark(mesh_level, worker_count, targets_hz)
    ref_json = _sequential_reference_json(mesh_level, targets_hz)

    payload: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "B3_Lmid_ST_multi_target_worker_scaling_benchmark_only",
        "B3_ST_scaling_mesh_level": mesh_level,
        "B3_ST_scaling_worker_count": worker_count,
        "B3_ST_scaling_targets_hz": list(targets_hz),
        "B3_ST_scaling_checkpoint_dir": str(checkpoint.resolve()),
        "B3_ST_scaling_operator_reuse_checkpoint_dir": str(reuse_ckpt) if reuse_ckpt else None,
        "B3_ST_scaling_operator_build_skipped": bool(reuse_ckpt is not None),
        "B3_ST_scaling_output_json": str(out_json.resolve()),
        "B3_ST_scaling_in_process_parity": bool(worker_count == 1),
        "B3_ST_scaling_solver_name": "KRYLOVSCHUR-ST-SINVERT-MUMPS-MULTI-TARGET-WORKER-SCALING",
        "production_promotion": "BLOCKED",
        "no_automatic_production_promotion": True,
        "B3_ST_scaling_recommended_parent_launch": "plain_python_not_mpiexec",
        "B3_ST_scaling_parent_mpi_world_size": int(mpi_size),
        "B3_ST_scaling_parent_inherited_launch_env_key_count": len(inherited_launch_env),
    }
    if inherited_launch_env:
        payload["B3_ST_scaling_parent_inherited_launch_env_keys_sample"] = inherited_launch_env[:32]
    timer = dev_bench._B3DevTiming(payload)
    op_profile = B3OperatorBuildProfiler.maybe_from_env(payload)
    mats: List[Any] = []
    seen: set[int] = set()
    verdict = "B3_ST_WORKER_SCALING_BLOCKED"
    rc = 2

    loop_summary: Dict[str, Any] = {}
    try:
        if reuse_ckpt is not None:
            export_pass, export_missing, export_detail = _verify_operator_export(checkpoint)
            payload["export_matrices_pass"] = bool(export_pass)
            payload["export_matrices_detail"] = export_detail
            if not export_pass:
                payload["failure_reason"] = f"reuse_checkpoint_incomplete:{export_missing}"
                verdict = "B3_ST_WORKER_SCALING_REUSE_CHECKPOINT_BLOCKED"
                audit._write_json_atomic(out_json, payload)
                return 2
            built, meta, reuse_diag = _load_operators(checkpoint)
            mats.extend([built["A_active"], built["M_active"]])
            payload.update(reuse_diag)
            if not reuse_diag.get("B3_ST_scaling_reuse_checkpoint_metadata_schema_pass"):
                payload["failure_reason"] = reuse_diag.get(
                    "B3_ST_scaling_reuse_checkpoint_failure_reason"
                )
                verdict = "B3_ST_WORKER_SCALING_REUSE_CHECKPOINT_METADATA_BLOCKED"
                audit._write_json_atomic(out_json, payload)
                return 2
            payload["B3_ST_scaling_operator_build_elapsed_seconds"] = None
            payload["B3_ST_scaling_reused_active_dimension"] = int(meta.get("active_dimension", 0))
            op_payload: Dict[str, Any] = {}
            op_profile.begin("operator_contract")
            contract_pass = _st_scaling_operator_contract_pass(op_payload, built=built, mesh_level=mesh_level)
            op_profile.end("operator_contract")
            payload.update(op_payload)
            payload["B3_ST_scaling_operator_contract_pass"] = bool(contract_pass)
            if not contract_pass:
                payload["failure_reason"] = "operator_contract_failed_on_reused_checkpoint"
                verdict = "B3_ST_WORKER_SCALING_OPERATOR_CONTRACT_BLOCKED"
                _finalize_operator_build_profile(op_profile, payload=payload, mesh_level=mesh_level)
                audit._write_json_atomic(out_json, payload)
                return 2
            payload["B3_ST_scaling_operator_export"] = {"reused_checkpoint_dir": str(checkpoint.resolve())}
        else:
            timer.mark("operator_build_begin")
            built = audit._b3_build_corrected_structural_active_operators(
                mats_to_destroy=mats,
                mat_destroy_seen=seen,
                mesh_level=mesh_level,
                struct_active_count_policy=_struct_active_count_policy(mesh_level),
                operator_build_profile=op_profile,
            )
            op_profile = built.get("operator_build_profile", op_profile)
            timer.mark("operator_build_end")
            payload["B3_ST_scaling_operator_build_elapsed_seconds"] = payload.get(
                "B3_DEV_timing_operator_build_end_elapsed_seconds"
            )

            op_payload = {}
            op_profile.begin("operator_contract")
            contract_pass = _st_scaling_operator_contract_pass(op_payload, built=built, mesh_level=mesh_level)
            op_profile.end("operator_contract")
            payload.update(op_payload)
            payload["B3_ST_scaling_operator_contract_pass"] = bool(contract_pass)
            if not contract_pass:
                payload["failure_reason"] = "operator_contract_failed"
                verdict = "B3_ST_WORKER_SCALING_OPERATOR_CONTRACT_BLOCKED"
                _finalize_operator_build_profile(op_profile, payload=payload, mesh_level=mesh_level)
                audit._write_json_atomic(out_json, payload)
                return 2

            op_profile.begin("checkpoint_export")
            export_meta = _export_operators(checkpoint, built=built, mesh_level=mesh_level)
            op_profile.end("checkpoint_export")
            payload["B3_ST_scaling_operator_export"] = export_meta
            export_pass, export_missing, export_detail = _verify_operator_export(checkpoint)
            payload["export_matrices_pass"] = bool(export_pass)
            payload["export_matrices_detail"] = export_detail
            if not export_pass:
                payload["failure_reason"] = f"operator_export_incomplete:{export_missing}"
                verdict = "B3_ST_WORKER_SCALING_EXPORT_BLOCKED"
                _finalize_operator_build_profile(op_profile, payload=payload, mesh_level=mesh_level)
                audit._write_json_atomic(out_json, payload)
                return 2
            _finalize_operator_build_profile(op_profile, payload=payload, mesh_level=mesh_level)

        shards = _partition_target_shards(targets_hz, worker_count)
        payload["B3_ST_scaling_target_shards"] = [
            {"worker_id": wi, "target_indices": [ti for ti, _ in ent], "target_frequencies_hz": [hz for _, hz in ent]}
            for wi, ent in enumerate(shards)
        ]

        wall_t0 = time.perf_counter()
        all_accepted: List[Dict[str, Any]] = []
        per_target_merged: Dict[str, Any] = {}
        total_setup_s = 0.0
        total_solve_s = 0.0
        worker_peak_rss: List[Optional[float]] = []
        worker_rcs: List[Optional[int]] = []

        if worker_count == 1:
            entries = shards[0]
            payload["B3_ST_scaling_execution_path"] = (
                "in_process_single_worker_with_reused_checkpoint"
                if reuse_ckpt
                else "in_process_single_worker_parity"
            )
            payload["worker_processes_pass"] = True
            accepted, per_t, setup_s, solve_s, loop_summary = _run_st_targets_on_built(
                built,
                entries,
                mesh_level=mesh_level,
                payload_prefix="B3_ST_scaling",
            )
            all_accepted.extend(accepted)
            per_target_merged.update(per_t)
            total_setup_s += setup_s
            total_solve_s += solve_s
            worker_peak_rss.append(_peak_rss_mb())
            worker_rcs.append(0)
        else:
            payload["B3_ST_scaling_execution_path"] = "plain_python_subprocess_sharded_workers"
            payload["B3_ST_scaling_worker_launch"] = "plain_python_no_nested_mpiexec"
            worker_specs = _write_worker_job_files(
                checkpoint, shards=shards, worker_count=worker_count, mesh_level=mesh_level
            )
            payload["B3_ST_scaling_worker_job_files_written"] = [
                spec["paths"]["job"].name for spec in worker_specs
            ]

            worker_return_code_map: Dict[int, int] = {}
            for spec in worker_specs:
                wi = int(spec["worker_id"])
                entries = spec["entries"]
                paths = spec["paths"]
                if not entries:
                    worker_rcs.append(None)
                    worker_peak_rss.append(None)
                    payload[f"B3_ST_scaling_worker_{wi}_skipped"] = True
                    continue
                w_rc = _spawn_worker_shard_plain_python(paths["job"], paths["log"])
                worker_return_code_map[wi] = int(w_rc)
                worker_rcs.append(int(w_rc))
                payload[f"B3_ST_scaling_worker_{wi}_return_code"] = int(w_rc)
                if int(w_rc) != 0:
                    _ensure_worker_failure_artifact(
                        paths,
                        worker_id=wi,
                        worker_count=worker_count,
                        return_code=int(w_rc),
                        reason=f"subprocess_exit_code_{w_rc}",
                    )
                if paths["log"].is_file():
                    try:
                        log_tail = paths["log"].read_text(encoding="utf-8", errors="replace").strip()[-4096:]
                        if log_tail:
                            payload[f"B3_ST_scaling_worker_{wi}_log_tail"] = log_tail
                    except OSError:
                        pass

            processes_pass, process_status = _collect_worker_process_status(
                checkpoint,
                worker_specs,
                worker_return_codes=worker_return_code_map,
            )
            payload.update(process_status)
            if not processes_pass:
                payload["failure_reason"] = (
                    process_status.get("worker_processes_failure_reason")
                    or f"worker_processes_missing:{process_status.get('worker_processes_missing_files')}"
                )
                verdict = "B3_ST_WORKER_SCALING_WORKER_FAILED"
                timer.finalize()
                payload["next_step_verdict"] = verdict
                audit._write_json_atomic(out_json, payload)
                return 2

            for spec in worker_specs:
                entries = spec["entries"]
                if not entries:
                    continue
                wi = int(spec["worker_id"])
                paths = spec["paths"]
                shard_path = paths["shard"]
                if not shard_path.is_file():
                    continue
                shard_data = json.loads(shard_path.read_text(encoding="utf-8"))
                worker_peak_rss.append(shard_data.get("B3_ST_scaling_shard_peak_rss_mb"))
                all_accepted.extend(shard_data.get("B3_ST_scaling_shard_accepted_mode_records") or [])
                for k, v in shard_data.items():
                    if k.startswith("B3_ST_scaling_target_"):
                        per_target_merged[k] = v
                total_setup_s += float(shard_data.get("B3_ST_scaling_shard_total_setup_elapsed_seconds") or 0.0)
                total_solve_s += float(shard_data.get("B3_ST_scaling_shard_total_solve_elapsed_seconds") or 0.0)
            loop_summary = _derive_loop_summary_from_per_target(per_target_merged, targets_hz)

        wall_elapsed = time.perf_counter() - wall_t0
        payload["B3_ST_scaling_st_phase_wall_elapsed_seconds"] = dev_bench._safe_float(wall_elapsed)
        payload["B3_ST_scaling_total_setup_elapsed_seconds"] = dev_bench._safe_float(total_setup_s)
        payload["B3_ST_scaling_total_solve_elapsed_seconds"] = dev_bench._safe_float(total_solve_s)
        payload.update(per_target_merged)

        union_freqs, provenance = _aggregate_union_and_provenance(all_accepted)
        payload["B3_ST_scaling_unique_accepted_frequency_count"] = len(union_freqs)
        payload["B3_ST_scaling_unique_accepted_frequencies"] = union_freqs
        payload["B3_ST_scaling_deduplicated_mode_provenance"] = provenance
        payload["B3_ST_scaling_per_target_accepted_mode_records"] = list(all_accepted)
        payload["B3_ST_scaling_worker_return_codes"] = worker_rcs
        rss_vals = [float(x) for x in worker_peak_rss if x is not None and math.isfinite(float(x))]
        payload["B3_ST_scaling_parent_peak_rss_mb"] = _peak_rss_mb()
        payload["B3_ST_scaling_worker_peak_rss_mb"] = worker_peak_rss
        payload["B3_ST_scaling_max_worker_peak_rss_mb"] = (
            dev_bench._safe_float(max(rss_vals)) if rss_vals else None
        )

        payload.update(_compare_to_sequential_reference(union_freqs, ref_path=ref_json, targets_hz=targets_hz))
        _record_solve_loop_status(payload, loop_summary=loop_summary, targets_hz=targets_hz)

        timer.finalize()
        payload["B3_ST_scaling_total_wall_elapsed_seconds"] = payload.get("B3_DEV_timing_total_wall_elapsed_seconds")

        verdict, rc = _finalize_scaling_verdict(payload, targets_hz=targets_hz)

    except Exception as exc:
        payload["failure_reason"] = f"{type(exc).__name__}:{exc}"
        verdict = "B3_ST_WORKER_SCALING_FAILED"
        rc = 2
    finally:
        audit._destroy_mats_deduped(mats)
        payload["next_step_verdict"] = verdict
        audit._write_json_atomic(out_json, payload)
        md_path = out_json.with_suffix(".md")
        md_path.write_text(
            f"# ST worker scaling benchmark ({payload.get('B3_ST_scaling_mesh_level')})\n\n"
            f"- mesh: `{payload.get('B3_ST_scaling_mesh_level')}`\n"
            f"- mesh_path: `{payload.get('B3_ST_scaling_mesh_path')}`\n"
            f"- active_dimension: `{payload.get('B3_ST_scaling_active_dimension')}`\n"
            f"- A_shape: `{payload.get('B3_ST_scaling_A_shape')}`\n"
            f"- M_shape: `{payload.get('B3_ST_scaling_M_shape')}`\n"
            f"- operator_build_s: `{payload.get('B3_ST_scaling_operator_build_elapsed_seconds')}`\n"
            f"- unique_accepted_frequency_count: `{payload.get('B3_ST_scaling_unique_accepted_frequency_count')}`\n"
            f"- targets_attempted: `{payload.get('B3_ST_scaling_targets_attempted_count')}`\n"
            f"- targets_setup_succeeded: `{payload.get('B3_ST_scaling_targets_setup_succeeded_count')}`\n"
            f"- targets_solved: `{payload.get('B3_ST_scaling_targets_solved_count')}`\n"
            f"- solve_loop_entered: `{payload.get('B3_ST_scaling_solve_loop_entered_pass')}`\n"
            f"- workers: `{payload.get('B3_ST_scaling_worker_count')}`\n"
            f"- peak_rss_mb: `{payload.get('B3_ST_scaling_parent_peak_rss_mb')}`\n"
            f"- verdict: `{verdict}`\n"
            f"- json: `{out_json}`\n",
            encoding="utf-8",
        )
        print(f"[B3_ST_scaling] written={out_json} verdict={verdict}", flush=True)
    return rc
