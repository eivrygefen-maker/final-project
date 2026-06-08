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

from v2_b3_checkpoint_metadata_lib import normalize_checkpoint_metadata
from v2_b3_petsc_util import mat_shape, petsc_mat_try_assemble, write_json_atomic

ACCEPTANCE_FREQ_LO_HZ = 220.0
ACCEPTANCE_FREQ_HI_HZ = 265.0
FACTOR_SHIFT_AMOUNT = 1.0e-8
L_PROD_ST_FULL9_TARGETS_HZ = [221.5, 227.0, 232.5, 238.0, 243.5, 249.0, 254.5, 260.0, 264.0]
FREQ_PARITY_TOL_HZ = 0.05

ACCEPTANCE_POLICY_LEGACY = "legacy_global_interval"
ACCEPTANCE_POLICY_DISCOVERY = "discovery_band_and_target_window"


class AcceptanceConfig:
    """Mode frequency acceptance policy for checkpoint ST solves."""

    def __init__(
        self,
        *,
        policy: str = ACCEPTANCE_POLICY_LEGACY,
        freq_lo: float = ACCEPTANCE_FREQ_LO_HZ,
        freq_hi: float = ACCEPTANCE_FREQ_HI_HZ,
        discovery_band_hz: Optional[Tuple[float, float]] = None,
        target_window_half_width_hz: Optional[float] = None,
        per_target_windows_hz: Optional[Dict[float, Tuple[float, float]]] = None,
    ) -> None:
        self.policy = str(policy)
        self.freq_lo = float(freq_lo)
        self.freq_hi = float(freq_hi)
        self.discovery_band_hz = discovery_band_hz
        self.target_window_half_width_hz = (
            float(target_window_half_width_hz)
            if target_window_half_width_hz is not None
            else None
        )
        self.per_target_windows_hz = dict(per_target_windows_hz) if per_target_windows_hz else None

    @classmethod
    def legacy(cls) -> AcceptanceConfig:
        return cls(policy=ACCEPTANCE_POLICY_LEGACY)

    @classmethod
    def discovery(
        cls,
        *,
        band_lo_hz: float,
        band_hi_hz: float,
        target_window_half_width_hz: float,
    ) -> AcceptanceConfig:
        if band_hi_hz <= band_lo_hz:
            raise ValueError(f"discovery band invalid: [{band_lo_hz}, {band_hi_hz}]")
        if target_window_half_width_hz <= 0.0:
            raise ValueError(
                f"target_window_half_width_hz must be positive, got {target_window_half_width_hz}"
            )
        return cls(
            policy=ACCEPTANCE_POLICY_DISCOVERY,
            discovery_band_hz=(float(band_lo_hz), float(band_hi_hz)),
            target_window_half_width_hz=float(target_window_half_width_hz),
        )

    @property
    def discovery_mode(self) -> bool:
        return self.policy == ACCEPTANCE_POLICY_DISCOVERY

    def mode_frequency_inside(self, f_hz: float, *, target_hz: float) -> bool:
        if f_hz is None:
            return False
        f = float(f_hz)
        if self.discovery_mode:
            if self.discovery_band_hz is None:
                raise RuntimeError("discovery policy missing band")
            lo, hi = self.discovery_band_hz
            if not (lo <= f <= hi):
                return False
            t = float(target_hz)
            if self.per_target_windows_hz:
                win = self.per_target_windows_hz.get(t)
                if win is None:
                    for k, v in self.per_target_windows_hz.items():
                        if abs(float(k) - t) < 1.0e-4:
                            win = v
                            break
                if win is not None:
                    return float(win[0]) <= f <= float(win[1]) + 1.0e-9
            if self.target_window_half_width_hz is None:
                raise RuntimeError("discovery policy missing window half-width")
            half = float(self.target_window_half_width_hz)
            return abs(f - t) <= half + 1.0e-9
        return self.freq_lo <= f <= self.freq_hi

    def per_target_window_hz(self, target_hz: float) -> Optional[List[float]]:
        if not self.discovery_mode:
            return None
        t = float(target_hz)
        if self.per_target_windows_hz:
            win = self.per_target_windows_hz.get(t)
            if win is None:
                for k, v in self.per_target_windows_hz.items():
                    if abs(float(k) - t) < 1.0e-4:
                        win = v
                        break
            if win is not None:
                return [float(win[0]), float(win[1])]
        if self.target_window_half_width_hz is None:
            return None
        half = float(self.target_window_half_width_hz)
        return [t - half, t + half]

    def to_result_fields(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "discovery_mode": bool(self.discovery_mode),
            "accepted_frequency_policy": self.policy,
            "legacy_acceptance_interval_hz": [ACCEPTANCE_FREQ_LO_HZ, ACCEPTANCE_FREQ_HI_HZ],
            "acceptance_interval_hz": [self.freq_lo, self.freq_hi],
            "dedupe_tolerance_hz": FREQ_PARITY_TOL_HZ,
        }
        if self.discovery_mode and self.discovery_band_hz is not None:
            out["discovery_band_hz"] = list(self.discovery_band_hz)
            out["target_window_half_width_hz"] = self.target_window_half_width_hz
            out["acceptance_interval_hz"] = list(self.discovery_band_hz)
            out["per_target_windows_from_plan"] = bool(self.per_target_windows_hz)
        return out


def resolve_acceptance_config(
    *,
    discovery_mode: bool = False,
    discovery_band_hz: Optional[Sequence[float]] = None,
    target_window_half_width_hz: Optional[float] = None,
) -> AcceptanceConfig:
    if not discovery_mode:
        return AcceptanceConfig.legacy()
    if discovery_band_hz is None or len(discovery_band_hz) != 2:
        raise ValueError("--B3-discovery-mode requires --discovery-band-hz LO HI")
    if target_window_half_width_hz is None:
        raise ValueError("--B3-discovery-mode requires --target-window-half-width-hz")
    return AcceptanceConfig.discovery(
        band_lo_hz=float(discovery_band_hz[0]),
        band_hi_hz=float(discovery_band_hz[1]),
        target_window_half_width_hz=float(target_window_half_width_hz),
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
    acceptance_config: Optional[AcceptanceConfig] = None,
    freq_lo: float = ACCEPTANCE_FREQ_LO_HZ,
    freq_hi: float = ACCEPTANCE_FREQ_HI_HZ,
    export_vectors: bool = False,
    region_ctx: Optional[Dict[str, Any]] = None,
) -> Tuple[int, List[Dict[str, Any]]]:
    from slepc4py import SLEPc

    cfg = acceptance_config or AcceptanceConfig(
        policy=ACCEPTANCE_POLICY_LEGACY,
        freq_lo=float(freq_lo),
        freq_hi=float(freq_hi),
    )

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
            inside = cfg.mode_frequency_inside(f_hz, target_hz=float(target_hz))
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
                entry: Dict[str, Any] = {
                    "mode_index": i,
                    "eps_slot_index": int(i),
                    "frequency_hz": float(f_hz),
                    "lambda_real": lam_re,
                    "lambda_imag": lam_im,
                    "eps_compute_error_relative": safe_float(eps_err),
                    "st_shift_target_hz": float(target_hz),
                    "u_norm_W": safe_float(u_norm),
                    "p_norm_W": safe_float(p_norm),
                    "x_norm_W": safe_float(x_norm),
                    "p_support": safe_float(p_support),
                }
                if export_vectors:
                    entry["x_active"] = x_active.copy()
                try:
                    from v2_b3_mode_region_participation import attach_participation_to_accepted_mode

                    attach_participation_to_accepted_mode(
                        entry,
                        x_active=x_active,
                        built=built,
                        region_ctx=region_ctx,
                    )
                except Exception as exc:
                    entry["dominant_region"] = "unknown"
                    entry["top_participation"] = None
                    entry["back_participation"] = None
                    entry["air_participation"] = None
                    entry["participation_method"] = "not_available"
                    entry["participation_status"] = "not_available"
                    entry["participation_detail"] = f"participation_attach_failed:{type(exc).__name__}"
                try:
                    from v2_b3_mode_audio_coupling import attach_audio_coupling_to_accepted_mode

                    attach_audio_coupling_to_accepted_mode(
                        entry,
                        x_active=x_active,
                        built=built,
                        region_ctx=region_ctx,
                    )
                except Exception as exc:
                    entry["audio_coupling_status"] = "not_available"
                    entry["audio_coupling_method"] = "lightweight_modal_coupling_v1"
                    entry["audio_coupling_detail"] = f"audio_coupling_attach_failed:{type(exc).__name__}"
                try:
                    from v2_b3_m4_mode_provenance import attach_mode_provenance

                    attach_mode_provenance(
                        entry,
                        x_active=x_active,
                        built=built,
                        region_ctx=region_ctx,
                        solver_backend="mkl_pardiso",
                        solver_fallback_used=False,
                    )
                except Exception as exc:
                    entry["provenance_attach_error"] = f"{type(exc).__name__}:{exc}"
                accepted.append(entry)
        finally:
            vr.destroy()
            vi.destroy()
    return nconv, accepted


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


def petsc_version_query() -> Any:
    try:
        return PETSc.Sys.getVersion()
    except Exception:
        return "unknown"


def slepc_version_query() -> Any:
    try:
        from slepc4py import SLEPc
    except Exception:
        return "unknown"
    try:
        if hasattr(SLEPc, "GetVersion"):
            return SLEPc.GetVersion()
        sys_mod = getattr(SLEPc, "Sys", None)
        if sys_mod is not None and hasattr(sys_mod, "getVersion"):
            return sys_mod.getVersion()
    except Exception:
        pass
    return "unknown"


def version_snapshot() -> Dict[str, Any]:
    return {
        "petsc_version": petsc_version_query(),
        "slepc_version": slepc_version_query(),
    }


def parse_hz_list(text: str) -> List[float]:
    out: List[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        v = float(part)
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError(f"invalid_target_frequency_hz:{part}")
        out.append(v)
    if not out:
        raise ValueError("empty_target_frequency_list")
    return out


def deduplicate_frequencies_hz(freqs: List[float], *, tol_hz: float = FREQ_PARITY_TOL_HZ) -> List[float]:
    if not freqs:
        return []
    sorted_f = sorted(float(f) for f in freqs)
    out = [sorted_f[0]]
    for f in sorted_f[1:]:
        if abs(f - out[-1]) > tol_hz:
            out.append(f)
    return out


def freq_lists_match(a: List[float], b: List[float], *, tol_hz: float = FREQ_PARITY_TOL_HZ) -> bool:
    if len(a) != len(b):
        return False
    aa = sorted(float(x) for x in a)
    bb = sorted(float(x) for x in b)
    return all(abs(x - y) <= tol_hz for x, y in zip(aa, bb))


def run_checkpoint_st_target(
    *,
    A_active: Any,
    M_active: Any,
    built: Dict[str, Any],
    target_hz: float,
    factor_solver: str,
    mesh_level: Optional[str],
    nev: int,
    ncv: int,
    target_index: Optional[int] = None,
    export_vectors: bool = False,
    acceptance_config: Optional[AcceptanceConfig] = None,
    region_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one EPSSetUp + EPSSolve for a loaded checkpoint (solver-only)."""
    from slepc4py import SLEPc

    cfg = acceptance_config or AcceptanceConfig.legacy()
    target_lambda = float(hz_to_lambda_sq(float(target_hz)))
    fs = str(factor_solver).strip().lower()
    result: Dict[str, Any] = {
        "target_index": target_index,
        "target_frequency_hz": float(target_hz),
        "target_lambda": safe_float(target_lambda),
        "factor_solver": fs,
        "nev": int(nev),
        "ncv": int(ncv),
        "setup_succeeded": False,
        "solve_succeeded": False,
        "setup_elapsed_seconds": None,
        "solve_elapsed_seconds": None,
        "st_total_elapsed_seconds": None,
        "peak_rss_mb": None,
        "converged_mode_count": None,
        "converged_modes": [],
        "accepted_mode_count_in_interval": None,
        "accepted_frequencies_hz": [],
        "accepted_modes": [],
        "factor_solver_effective": None,
        "mumps_policy_effective": None,
        "mumps_policies_tried": [],
        "petsc_options_written": None,
        "configure_meta": None,
        "status": "FAIL",
        "failure_reason": None,
        "failure_class": None,
    }
    result.update(cfg.to_result_fields())
    win = cfg.per_target_window_hz(float(target_hz))
    if win is not None:
        result["per_target_acceptance_window_hz"] = win

    if fs == "mumps":
        policies = mumps_policy_chain(mesh_level=mesh_level)
    else:
        policies = [None]

    eps = None
    t_st0 = time.perf_counter()
    setup_succeeded = False
    last_setup_exc: Optional[BaseException] = None

    try:
        for policy in policies:
            if eps is not None:
                try:
                    eps.destroy()
                except Exception:
                    pass
                eps = None
            if policy is not None:
                result["mumps_policies_tried"].append(str(policy))
            try:
                eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
                configure_meta = configure_eps_krylovschur_sinvert(
                    eps,
                    A_active,
                    M_active,
                    target_hz=float(target_hz),
                    target_lambda=target_lambda,
                    factor_solver=fs,
                    nev=int(nev),
                    ncv=int(ncv),
                    mumps_policy=policy,
                )
                t0 = time.perf_counter()
                eps.setUp()
                setup_s = time.perf_counter() - t0
                result["setup_elapsed_seconds"] = safe_float(setup_s)
                result["setup_succeeded"] = True
                result["configure_meta"] = configure_meta
                result["factor_solver_effective"] = configure_meta.get("factor_solver_effective")
                result["mumps_policy_effective"] = configure_meta.get("mumps_policy_applied")
                result["petsc_options_written"] = configure_meta.get("petsc_options_written")
                setup_succeeded = True
                break
            except Exception as exc:
                last_setup_exc = exc
                diag = extract_st_failure_diagnostics(exc)
                result["failure_class"] = diag.get("failure_class")
                result["failure_reason"] = f"{type(exc).__name__}:{exc}"

        if not setup_succeeded:
            result["status"] = "FAIL_SETUP"
            if last_setup_exc is not None:
                result["failure_diagnostics"] = extract_st_failure_diagnostics(last_setup_exc)
            result["st_total_elapsed_seconds"] = safe_float(time.perf_counter() - t_st0)
            return result

        t0 = time.perf_counter()
        try:
            eps.solve()
            solve_s = time.perf_counter() - t0
            result["solve_elapsed_seconds"] = safe_float(solve_s)
            result["solve_succeeded"] = True
        except Exception as exc:
            result["failure_reason"] = f"{type(exc).__name__}:{exc}"
            result["failure_class"] = extract_st_failure_diagnostics(exc).get("failure_class")
            result["status"] = "FAIL_SOLVE"
            result["st_total_elapsed_seconds"] = safe_float(time.perf_counter() - t_st0)
            return result

        nconv, converged_modes = collect_converged_modes(eps, A_active)
        _nconv2, accepted_modes = collect_accepted_st_modes(
            eps,
            A_active,
            built,
            target_hz=float(target_hz),
            acceptance_config=cfg,
            export_vectors=bool(export_vectors),
            region_ctx=region_ctx,
        )
        accepted_freqs = sorted(float(m["frequency_hz"]) for m in accepted_modes)
        result["converged_mode_count"] = int(nconv)
        result["converged_modes"] = converged_modes
        result["accepted_mode_count_in_interval"] = len(accepted_modes)
        result["accepted_modes"] = accepted_modes
        result["accepted_frequencies_hz"] = accepted_freqs
        result["peak_rss_mb"] = peak_rss_mb()
        result["st_total_elapsed_seconds"] = safe_float(time.perf_counter() - t_st0)
        result["status"] = "PASS"
        return result
    finally:
        if eps is not None:
            try:
                eps.destroy()
            except Exception:
                pass


def compare_checkpoint_results_to_baseline(
    *,
    current: Dict[str, Any],
    baseline: Dict[str, Any],
    tol_hz: float = FREQ_PARITY_TOL_HZ,
) -> Dict[str, Any]:
    """Compare multi- or single-target accepted frequencies against a baseline JSON."""
    def _aggregate_freqs(body: Dict[str, Any]) -> List[float]:
        agg = body.get("aggregate") or {}
        if agg.get("unique_accepted_frequencies_hz"):
            return list(agg["unique_accepted_frequencies_hz"])
        if body.get("accepted_frequencies_hz"):
            return deduplicate_frequencies_hz(list(body["accepted_frequencies_hz"]), tol_hz=tol_hz)
        freqs: List[float] = []
        for row in body.get("targets") or []:
            freqs.extend(list(row.get("accepted_frequencies_hz") or []))
        return deduplicate_frequencies_hz(freqs, tol_hz=tol_hz)

    cur_freqs = _aggregate_freqs(current)
    base_freqs = _aggregate_freqs(baseline)
    per_target: List[Dict[str, Any]] = []
    cur_targets = {float(r.get("target_frequency_hz")): r for r in (current.get("targets") or [])}
    base_targets = {float(r.get("target_frequency_hz")): r for r in (baseline.get("targets") or [])}
    if cur_targets and base_targets:
        for hz in sorted(set(cur_targets) | set(base_targets)):
            c_row = cur_targets.get(hz) or {}
            b_row = base_targets.get(hz) or {}
            c_acc = list(c_row.get("accepted_frequencies_hz") or [])
            b_acc = list(b_row.get("accepted_frequencies_hz") or [])
            per_target.append(
                {
                    "target_frequency_hz": hz,
                    "accepted_frequencies_match": freq_lists_match(c_acc, b_acc, tol_hz=tol_hz),
                    "baseline_accepted_frequencies_hz": b_acc,
                    "current_accepted_frequencies_hz": c_acc,
                }
            )
    return {
        "baseline_path": baseline.get("_baseline_path"),
        "aggregate_accepted_frequencies_match": freq_lists_match(cur_freqs, base_freqs, tol_hz=tol_hz),
        "baseline_unique_accepted_frequencies_hz": base_freqs,
        "current_unique_accepted_frequencies_hz": cur_freqs,
        "per_target": per_target,
        "parity_pass": bool(
            freq_lists_match(cur_freqs, base_freqs, tol_hz=tol_hz)
            and (not per_target or all(r.get("accepted_frequencies_match") for r in per_target))
        ),
    }


CHECKPOINT_SOLVER_SUMMARY_SCHEMA = "checkpoint_solver_summary_v1"


def _per_target_summary_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = list(result.get("targets") or [])
    if not rows and result.get("target_frequency_hz") is not None:
        rows = [result]
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "target_hz": safe_float(row.get("target_frequency_hz")),
                "setup_s": safe_float(row.get("setup_elapsed_seconds")),
                "solve_s": safe_float(row.get("solve_elapsed_seconds")),
                "st_total_s": safe_float(row.get("st_total_elapsed_seconds")),
                "accepted_n": int(row.get("accepted_mode_count_in_interval") or 0),
                "status": row.get("status"),
            }
        )
    return out


def build_stable_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """Stable top-level summary block for result.json consumers."""
    agg = result.get("aggregate") or {}
    per_target = _per_target_summary_rows(result)
    targets_total = int(agg.get("targets_attempted") or len(per_target) or 0)
    targets_succeeded = int(agg.get("targets_succeeded") or sum(1 for r in per_target if r.get("status") == "PASS"))

    all_accepted: List[float] = []
    if agg.get("unique_accepted_frequencies_hz"):
        all_accepted = [float(x) for x in agg["unique_accepted_frequencies_hz"]]
    elif result.get("accepted_frequencies_hz"):
        all_accepted = [float(x) for x in result["accepted_frequencies_hz"]]
    else:
        for row in result.get("targets") or []:
            all_accepted.extend(float(x) for x in (row.get("accepted_frequencies_hz") or []))
    unique_accepted = deduplicate_frequencies_hz(all_accepted)

    aggregate_wall_s = safe_float(agg.get("total_wall_seconds") or result.get("total_elapsed_seconds"))
    total_st_s = safe_float(agg.get("total_st_seconds"))
    total_setup_s = safe_float(agg.get("total_setup_seconds"))
    total_solve_s = safe_float(agg.get("total_solve_seconds"))
    if total_st_s is None and len(per_target) == 1:
        total_st_s = per_target[0].get("st_total_s")
        total_setup_s = per_target[0].get("setup_s")
        total_solve_s = per_target[0].get("solve_s")
    if total_st_s is None and per_target:
        total_st_s = safe_float(sum(float(r.get("st_total_s") or 0.0) for r in per_target))
        total_setup_s = safe_float(sum(float(r.get("setup_s") or 0.0) for r in per_target))
        total_solve_s = safe_float(sum(float(r.get("solve_s") or 0.0) for r in per_target))

    return {
        "schema": CHECKPOINT_SOLVER_SUMMARY_SCHEMA,
        "status": result.get("status"),
        "factor_solver": result.get("factor_solver"),
        "aggregate_wall_s": aggregate_wall_s,
        "total_st_s": total_st_s,
        "total_setup_s": total_setup_s,
        "total_solve_s": total_solve_s,
        "targets_succeeded": targets_succeeded,
        "targets_total": targets_total,
        "unique_accepted_hz": unique_accepted,
        "per_target": per_target,
    }


def extract_summary_view(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return stable summary; rebuild from legacy keys when absent."""
    summary = result.get("summary")
    if isinstance(summary, dict) and summary.get("schema") == CHECKPOINT_SOLVER_SUMMARY_SCHEMA:
        return summary
    return build_stable_summary(result)


def compare_checkpoint_summaries(
    *,
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    tol_hz: float = FREQ_PARITY_TOL_HZ,
) -> Dict[str, Any]:
    """Compare two benchmark result.json files using stable summary fields."""
    base = extract_summary_view(baseline)
    cand = extract_summary_view(candidate)

    def _speedup(base_s: Any, cand_s: Any) -> Optional[float]:
        try:
            b = float(base_s)
            c = float(cand_s)
        except (TypeError, ValueError):
            return None
        if c <= 0.0:
            return None
        return b / c

    base_freqs = list(base.get("unique_accepted_hz") or [])
    cand_freqs = list(cand.get("unique_accepted_hz") or [])
    timing = {
        "aggregate_wall_speedup": _speedup(base.get("aggregate_wall_s"), cand.get("aggregate_wall_s")),
        "total_st_speedup": _speedup(base.get("total_st_s"), cand.get("total_st_s")),
        "total_setup_speedup": _speedup(base.get("total_setup_s"), cand.get("total_setup_s")),
        "total_solve_speedup": _speedup(base.get("total_solve_s"), cand.get("total_solve_s")),
    }
    per_target_cmp: List[Dict[str, Any]] = []
    base_by_hz = {float(r["target_hz"]): r for r in (base.get("per_target") or []) if r.get("target_hz") is not None}
    cand_by_hz = {float(r["target_hz"]): r for r in (cand.get("per_target") or []) if r.get("target_hz") is not None}
    for hz in sorted(set(base_by_hz) | set(cand_by_hz)):
        b_row = base_by_hz.get(hz) or {}
        c_row = cand_by_hz.get(hz) or {}
        per_target_cmp.append(
            {
                "target_hz": hz,
                "baseline_st_total_s": b_row.get("st_total_s"),
                "candidate_st_total_s": c_row.get("st_total_s"),
                "st_total_speedup": _speedup(b_row.get("st_total_s"), c_row.get("st_total_s")),
            }
        )

    return {
        "baseline_factor_solver": base.get("factor_solver"),
        "candidate_factor_solver": cand.get("factor_solver"),
        "baseline_status": base.get("status"),
        "candidate_status": cand.get("status"),
        "timing": timing,
        "baseline_unique_accepted_hz": base_freqs,
        "candidate_unique_accepted_hz": cand_freqs,
        "accepted_frequencies_match": freq_lists_match(base_freqs, cand_freqs, tol_hz=tol_hz),
        "per_target_timing": per_target_cmp,
        "parity_pass": bool(
            base.get("status") == "PASS"
            and cand.get("status") == "PASS"
            and freq_lists_match(base_freqs, cand_freqs, tol_hz=tol_hz)
        ),
    }
