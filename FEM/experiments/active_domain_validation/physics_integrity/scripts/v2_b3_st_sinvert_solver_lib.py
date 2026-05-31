#!/usr/bin/env python3
"""FEM-free KRYLOVSCHUR + ST.SINVERT configuration and mode acceptance."""
from __future__ import annotations

import json
import math
import os
import resource
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from petsc4py import PETSc

from v2_b3_petsc_util import mat_shape, petsc_mat_try_assemble, write_json_atomic

ACCEPTANCE_FREQ_LO_HZ = 220.0
ACCEPTANCE_FREQ_HI_HZ = 265.0
FACTOR_SHIFT_AMOUNT = 1.0e-8

_CHECKPOINT_METADATA_REQUIRED_KEYS = (
    "mesh_level",
    "active_dimension",
    "active_local",
    "inactive_local",
    "free_rows",
    "bc_rows",
    "u_idx",
    "p_idx",
    "n_w",
    "n_u_b3",
)


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def hz_to_lambda_sq(hz: float) -> float:
    f = max(float(hz), 0.0)
    return (2.0 * math.pi * f) ** 2


def lambda_hz_from_eigenvalue(lam_re: float, lam_im: float) -> Optional[float]:
    if not (math.isfinite(lam_re) and math.isfinite(lam_im)):
        return None
    if abs(lam_im) > 1.0e-12 or lam_re <= 0.0:
        return None
    return math.sqrt(max(lam_re, 0.0)) / (2.0 * math.pi)


def lambda_near_unity_signature(f_hz: Any, *, rtol: float = 1.0e-6) -> bool:
    if f_hz is None or isinstance(f_hz, str):
        return False
    try:
        f_v = float(f_hz)
    except Exception:
        return False
    if not math.isfinite(f_v):
        return False
    lam = (2.0 * math.pi * f_v) ** 2
    if not math.isfinite(lam):
        return False
    return abs(lam - 1.0) <= float(rtol) * max(1.0, abs(lam))


def peak_rss_mb() -> Optional[float]:
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        rss = float(getattr(ru, "ru_maxrss", 0.0))
        if rss <= 0.0:
            return None
        if rss > 1.0e9:
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0
    except Exception:
        return None


def mat_global_nnz_used(mat: Any) -> Optional[int]:
    try:
        info = mat.getInfo()
        if isinstance(info, dict):
            for key in ("nz_used", "nz_allocated"):
                if key in info:
                    return int(info[key])
    except Exception:
        pass
    try:
        petsc_mat_try_assemble(mat)
        _indptr, _indices, data = mat.getValuesCSR()
        return int(np.asarray(data).size)
    except Exception:
        return None


def pc_factor_solver_effective_label(pc: Any) -> Optional[str]:
    try:
        if hasattr(pc, "getFactorSolverType"):
            return str(pc.getFactorSolverType())
    except Exception:
        pass
    try:
        val = PETSc.Options().getString("st_pc_factor_mat_solver_type", "")
        if val:
            return str(val)
    except Exception:
        pass
    return None


def st_mumps_policy_spec(policy: str) -> Dict[str, Any]:
    base = {
        "st_mat_mumps_icntl_6": 7,
        "st_mat_mumps_icntl_12": 1,
        "st_mat_mumps_icntl_7": 0,
        "st_mat_mumps_icntl_4": 0,
    }
    if policy == "default":
        return {**base, "st_mat_mumps_icntl_14": 500, "st_mat_mumps_icntl_24": 0}
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
    if policy == "metis_ordering":
        return {**base, "st_mat_mumps_icntl_14": 800, "st_mat_mumps_icntl_24": 0, "st_mat_mumps_icntl_7": 5}
    raise ValueError(f"unknown_mumps_policy={policy}")


def mumps_policy_chain(*, mesh_level: Optional[str]) -> List[str]:
    if str(mesh_level or "") == "L_prod":
        return ["default", "L_prod_relaxed", "L_prod_maximum"]
    return ["default"]


def apply_st_mumps_petsc_policy(policy: str) -> Dict[str, Any]:
    spec = st_mumps_policy_spec(policy)
    petsc_opts = PETSc.Options()
    for key, val in spec.items():
        petsc_opts[key] = val
    return {"mumps_policy_applied": policy, "petsc_options_written": dict(spec)}


def direct_lu_solver_cfg(*, factor_solver: str) -> Dict[str, Any]:
    fs = str(factor_solver).strip().lower()
    return {
        "st_ksp_type": "preonly",
        "st_pc_type": "lu",
        "st_pc_factor_mat_solver_type": fs,
        "st_factor_solver_type": fs,
        "st_pc_factor_shift_type": "nonzero",
        "st_pc_factor_shift_amount": float(FACTOR_SHIFT_AMOUNT),
        "pc_factor_shift_type": "nonzero",
        "pc_factor_shift_amount": float(FACTOR_SHIFT_AMOUNT),
    }


def configure_st_ksp_pc_lu(
    ksp: Any,
    pc: Any,
    solver_cfg: Dict[str, Any],
    *,
    opts_prefix: str = "st_",
) -> Dict[str, Any]:
    """Configure ST inner KSP/PC for monolithic direct LU (no FieldSplit)."""
    ksp.setType(str(solver_cfg.get("st_ksp_type", "preonly")))
    pc.setType(str(solver_cfg.get("st_pc_type", "lu")))
    st_factor = str(
        solver_cfg.get(
            "st_pc_factor_mat_solver_type",
            solver_cfg.get("st_factor_solver_type", "mumps"),
        )
    )
    shift_type = str(solver_cfg.get("st_pc_factor_shift_type", "nonzero"))
    shift_amt = float(solver_cfg.get("st_pc_factor_shift_amount", FACTOR_SHIFT_AMOUNT))
    petsc_opts = PETSc.Options()
    petsc_opts[f"{opts_prefix}pc_type"] = "lu"
    petsc_opts[f"{opts_prefix}pc_factor_mat_solver_type"] = st_factor
    petsc_opts[f"{opts_prefix}pc_factor_shift_type"] = shift_type
    petsc_opts[f"{opts_prefix}pc_factor_shift_amount"] = shift_amt
    try:
        pc.setFactorSolverType(st_factor)
    except Exception:
        pass
    try:
        pc.setFactorShiftType(shift_type)
        pc.setFactorShiftAmount(shift_amt)
    except Exception:
        pass
    return {
        "ksp_type_effective": str(ksp.getType()),
        "pc_type_effective": str(pc.getType()),
        "factor_solver_requested": st_factor,
        "factor_shift_type": shift_type,
        "factor_shift_amount": shift_amt,
    }


def configure_eps_krylovschur_sinvert(
    eps: Any,
    A_active: Any,
    M_active: Any,
    *,
    target_hz: float,
    target_lambda: float,
    factor_solver: str,
    nev: int,
    ncv: int,
    mumps_policy: Optional[str] = None,
) -> Dict[str, Any]:
    from slepc4py import SLEPc

    fs = str(factor_solver).strip().lower()
    meta: Dict[str, Any] = {
        "target_frequency_hz": float(target_hz),
        "target_lambda": safe_float(target_lambda),
        "eps_type_requested": "KRYLOVSCHUR",
        "st_type_requested": "SINVERT",
        "problem_type_requested": "GNHEP",
        "which_requested": "TARGET_MAGNITUDE",
        "nev_requested": int(nev),
        "ncv_requested": int(ncv),
        "factor_solver_requested": fs,
    }

    eps.setOperators(A_active, M_active)
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.setTarget(float(target_lambda))
    try:
        eps.setDimensions(nev=int(nev), ncv=int(ncv))
    except TypeError:
        eps.setDimensions(int(nev), int(ncv))

    st = eps.getST()
    try:
        st.setType(SLEPc.ST.Type.SINVERT)
    except Exception:
        st.setType("sinvert")
    st.setShift(float(target_lambda))

    ksp = st.getKSP()
    pc = ksp.getPC()
    ksp_meta = configure_st_ksp_pc_lu(ksp, pc, direct_lu_solver_cfg(factor_solver=fs))
    meta.update(ksp_meta)
    meta["st_type_effective"] = str(st.getType())

    if fs == "mumps" and mumps_policy is not None:
        meta.update(apply_st_mumps_petsc_policy(str(mumps_policy)))

    meta["factor_solver_effective"] = pc_factor_solver_effective_label(pc)
    return meta


def extract_st_failure_diagnostics(exc: BaseException) -> Dict[str, Any]:
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
    }
    if "infog(1)=-13" in lower or "infog[1]=-13" in lower:
        out["mumps_infog1"] = -13
        out["failure_class"] = "MUMPS_NUMERICAL_FACTORIZATION_OR_MEMORY"
    if "error code 76" in lower or (ierr is not None and int(ierr) == 76):
        out["failure_class"] = "PETSC_PC_FACTOR_SETUp_FAILED"
    if "memory" in lower or "alloc" in lower:
        out["failure_class"] = "MEMORY_ALLOCATION_OR_MUMPS_WORKSPACE"
    return out


def collect_converged_modes(
    eps: Any,
    A_active: Any,
    *,
    freq_lo: float = ACCEPTANCE_FREQ_LO_HZ,
    freq_hi: float = ACCEPTANCE_FREQ_HI_HZ,
) -> Tuple[int, List[Dict[str, Any]]]:
    from slepc4py import SLEPc

    nconv = int(eps.getConverged())
    modes: List[Dict[str, Any]] = []
    for i in range(nconv):
        vr = A_active.createVecRight()
        vi = A_active.createVecRight()
        try:
            lam = eps.getEigenpair(i, vr, vi)
            lam_re = float(np.real(complex(lam)))
            lam_im = float(np.imag(complex(lam)))
            f_hz = lambda_hz_from_eigenvalue(lam_re, lam_im)
            eps_err = float("nan")
            try:
                eps_err = float(eps.computeError(i, SLEPc.EPS.ErrorType.RELATIVE))
            except Exception:
                pass
            modes.append(
                {
                    "mode_index": i,
                    "lambda_real": lam_re,
                    "lambda_imag": lam_im,
                    "frequency_hz": safe_float(f_hz),
                    "inside_acceptance_interval": bool(
                        f_hz is not None and freq_lo <= float(f_hz) <= freq_hi
                    ),
                    "eps_compute_error_relative": safe_float(eps_err),
                }
            )
        finally:
            vr.destroy()
            vi.destroy()
    return nconv, modes


def collect_accepted_st_modes(
    eps: Any,
    A_active: Any,
    built: Dict[str, Any],
    *,
    target_hz: float,
    freq_lo: float = ACCEPTANCE_FREQ_LO_HZ,
    freq_hi: float = ACCEPTANCE_FREQ_HI_HZ,
) -> Tuple[int, List[Dict[str, Any]]]:
    from slepc4py import SLEPc

    nconv = int(eps.getConverged())
    accepted: List[Dict[str, Any]] = []
    free_rows = np.asarray(built["free_rows"], dtype=np.int32).ravel()
    bc_rows = np.unique(np.asarray(built["bc_rows"], dtype=np.int32).ravel())
    active_local = np.asarray(built["active_local"], dtype=np.int32).ravel()
    inactive_local = np.asarray(built["inactive_local"], dtype=np.int32).ravel()
    u_idx = np.asarray(built["u_idx"], dtype=np.int32).ravel()
    p_idx = np.asarray(built["p_idx"], dtype=np.int32).ravel()
    n_w = int(built["n_w"])
    n_free = int(free_rows.size)

    for i in range(nconv):
        vr = A_active.createVecRight()
        vi = A_active.createVecRight()
        try:
            lam = eps.getEigenpair(i, vr, vi)
            lam_re = float(np.real(complex(lam)))
            lam_im = float(np.imag(complex(lam)))
            finite = bool(math.isfinite(lam_re) and math.isfinite(lam_im))
            f_hz = lambda_hz_from_eigenvalue(lam_re, lam_im)
            inside = bool(f_hz is not None and freq_lo <= float(f_hz) <= freq_hi)
            eps_err = float("nan")
            try:
                eps_err = float(eps.computeError(i, SLEPc.EPS.ErrorType.RELATIVE))
            except Exception:
                pass
            eps_ok = bool(math.isfinite(eps_err) and eps_err <= 1.0e-4)
            x_active = np.asarray(vr.getArray(readonly=True), dtype=np.float64).ravel().copy()
            x_free = np.zeros(n_free, dtype=np.float64)
            x_free[active_local] = x_active
            x_full = np.zeros(n_w, dtype=np.float64)
            x_full[free_rows] = x_free
            si_norm = float(np.linalg.norm(x_free[inactive_local])) if inactive_local.size else 0.0
            d_norm = float(np.linalg.norm(x_full[bc_rows])) if bc_rows.size else 0.0
            x_norm = float(np.linalg.norm(x_full))
            si_pass = bool(si_norm <= 1.0e-8 * max(1.0, x_norm))
            d_pass = bool(d_norm <= 1.0e-8 * max(1.0, x_norm))
            u_norm = float(np.linalg.norm(np.abs(x_full[u_idx])))
            p_norm = float(np.linalg.norm(np.abs(x_full[p_idx])))
            p_support = p_norm / max(x_norm, 1.0e-30)
            support_ok = bool(u_norm > 1.0e-8 and (p_support > 1.0e-6 or (u_norm > 1.0e-8 and p_norm <= 1.0e-8)))
            lambda_one = bool(
                lambda_near_unity_signature(f_hz)
                or (abs(lam_re - 1.0) <= 1.0e-6 and abs(lam_im) <= 1.0e-9)
            )
            nonfinite = bool(not finite or math.isinf(lam_re) or math.isinf(lam_im))
            mode_pass = bool(
                finite
                and f_hz is not None
                and float(f_hz) > 0.0
                and inside
                and eps_ok
                and si_pass
                and d_pass
                and not lambda_one
                and not nonfinite
                and support_ok
            )
            if mode_pass:
                accepted.append(
                    {
                        "mode_index": i,
                        "frequency_hz": float(f_hz),
                        "lambda_real": lam_re,
                        "lambda_imag": lam_im,
                        "eps_compute_error_relative": safe_float(eps_err),
                        "st_shift_target_hz": float(target_hz),
                    }
                )
        finally:
            vr.destroy()
            vi.destroy()
    return nconv, accepted


def normalize_checkpoint_metadata(meta: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], bool]:
    missing_required = [k for k in _CHECKPOINT_METADATA_REQUIRED_KEYS if k not in meta]
    if missing_required:
        return dict(meta), missing_required, False
    normalized: Dict[str, Any] = dict(meta)
    inactive_local = np.asarray(meta["inactive_local"], dtype=np.int32).ravel()
    inactive_n = int(inactive_local.size)
    for key, default in (
        ("inactive_structural_count", inactive_n),
        ("inactive_pressure_count", 0),
        ("inactive_aup_overlap_count", 0),
        ("aup_supported_count", 0),
        ("parent_raw_Auu_exact_zero_count", inactive_n),
        ("parent_raw_Auu_nonzero_count", 0),
    ):
        normalized.setdefault(key, default)
    active_dim = int(normalized.get("active_dimension", 0))
    active_local = np.asarray(normalized["active_local"], dtype=np.int32).ravel()
    schema_pass = bool(active_dim > 0 and int(active_local.size) == active_dim and inactive_n >= 0)
    return normalized, missing_required, schema_pass


def built_from_checkpoint_metadata(
    meta: Dict[str, Any],
    *,
    A_active: Any,
    M_active: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    normalized, missing_required, schema_pass = normalize_checkpoint_metadata(meta)
    inactive_local = np.asarray(normalized["inactive_local"], dtype=np.int32).ravel()
    cand = {
        "inactive_local": inactive_local,
        "inactive_structural_count": int(normalized.get("inactive_structural_count", inactive_local.size)),
        "inactive_pressure_count": int(normalized.get("inactive_pressure_count", 0)),
        "inactive_aup_overlap_count": int(normalized.get("inactive_aup_overlap_count", 0)),
        "aup_supported_count": int(normalized.get("aup_supported_count", 0)),
        "parent_raw_Auu_exact_zero_count": int(
            normalized.get("parent_raw_Auu_exact_zero_count", inactive_local.size)
        ),
        "parent_raw_Auu_nonzero_count": int(normalized.get("parent_raw_Auu_nonzero_count", 0)),
    }
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
        "mesh_level": normalized.get("mesh_level"),
    }
    diag = {
        "checkpoint_metadata_schema_pass": bool(schema_pass),
        "checkpoint_metadata_missing_required": list(missing_required),
    }
    return built, diag


def threading_env_snapshot() -> Dict[str, Optional[str]]:
    keys = ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "MKL_DYNAMIC", "PETSC_OPTIONS")
    return {k: os.environ.get(k) for k in keys}


def version_snapshot() -> Dict[str, Any]:
    from slepc4py import SLEPc

    return {
        "petsc_version": PETSc.Sys.getVersion(),
        "slepc_version": SLEPc.GetVersion(),
    }
