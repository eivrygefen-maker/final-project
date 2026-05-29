#!/usr/bin/env python3
"""Coarse-mesh B3 development solver benchmarks (smoke tests; not final validation)."""
from __future__ import annotations

import json
import math
import resource
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import run_v2_B3_trace_coupled_operator_and_seed_transfer_audit as audit

B3_DEV_MESH_VARIANT_ARG = "--B3-dev-mesh-variant"
B3_DEV_COARSE_CONTRACT_ARG = "--B3-dev-coarse-corrected-operator-contract-only"
B3_DEV_COARSE_CISS_ARG = "--B3-dev-coarse-ciss-direct-stable-benchmark-only"
B3_DEV_COARSE_ST_ARG = "--B3-dev-coarse-krylovschur-sinvert-benchmark-only"
B3_DEV_COMPARE_ARG = "--B3-dev-solver-benchmark-compare-only"

B3_DEV_DEFAULT_MESH_VARIANT = "L_dev_coarse"
B3_DEV_COARSE_NEV = 12
B3_DEV_COARSE_NCV = 24

OUT_JSON_DEV_CONTRACT = audit.CONV_DIAG / "v2_B3_DEV_coarse_corrected_operator_contract_only.json"
OUT_MD_DEV_CONTRACT = audit.CONV_DIAG / "v2_B3_DEV_coarse_corrected_operator_contract_only.md"
OUT_JSON_DEV_CISS = audit.CONV_DIAG / "v2_B3_DEV_coarse_ciss_direct_stable_benchmark_only.json"
OUT_MD_DEV_CISS = audit.CONV_DIAG / "v2_B3_DEV_coarse_ciss_direct_stable_benchmark_only.md"
OUT_JSON_DEV_ST = audit.CONV_DIAG / "v2_B3_DEV_coarse_krylovschur_sinvert_benchmark_only.json"
OUT_MD_DEV_ST = audit.CONV_DIAG / "v2_B3_DEV_coarse_krylovschur_sinvert_benchmark_only.md"
OUT_JSON_DEV_SUMMARY = audit.CONV_DIAG / "v2_B3_DEV_coarse_solver_benchmark_summary.json"
OUT_MD_DEV_SUMMARY = audit.CONV_DIAG / "v2_B3_DEV_coarse_solver_benchmark_summary.md"


class _B3DevTiming:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self.payload = payload
        self._t0 = time.perf_counter()
        self._marks: Dict[str, float] = {}

    def mark(self, name: str) -> None:
        self._marks[name] = time.perf_counter() - self._t0
        self.payload[f"B3_DEV_timing_{name}_elapsed_seconds"] = _safe_float(self._marks[name])

    def finalize(self) -> None:
        self.payload["B3_DEV_timing_total_wall_elapsed_seconds"] = _safe_float(time.perf_counter() - self._t0)
        try:
            ru = resource.getrusage(resource.RUSAGE_SELF)
            self.payload["B3_DEV_peak_memory_rss_mb"] = _safe_float(float(ru.ru_maxrss) / 1024.0)
        except Exception:
            self.payload["B3_DEV_peak_memory_rss_mb"] = None


def _safe_float(x: Any) -> Optional[float]:
    return audit._safe_float(x)


def _parse_dev_mesh_variant(argv: List[str]) -> str:
    for i, arg in enumerate(argv):
        if arg == B3_DEV_MESH_VARIANT_ARG and i + 1 < len(argv):
            return str(argv[i + 1]).strip()
        if arg.startswith(f"{B3_DEV_MESH_VARIANT_ARG}="):
            return str(arg.split("=", 1)[1]).strip()
    return B3_DEV_DEFAULT_MESH_VARIANT


def _dev_mesh_bootstrap_payload(mesh_variant: str) -> Dict[str, Any]:
    mesh_file = audit.mesh_path(mesh_variant, audit.CASE_ID)
    node_count = None
    element_count = None
    audit_json = audit.CONV_DIAG.parent / "mesh" / mesh_variant / f"{audit.CASE_ID}_mesh_audit.json"
    if audit_json.is_file():
        try:
            ad = json.loads(audit_json.read_text(encoding="utf-8"))
            node_count = ad.get("n_nodes")
            element_count = ad.get("n_tetrahedra")
        except Exception:
            pass
    return {
        "B3_DEV_mesh_variant": mesh_variant,
        "B3_DEV_mesh_is_solver_smoke_test_only": True,
        "B3_DEV_mesh_not_authorized_for_final_physics_validation": True,
        "B3_DEV_mesh_path": str(mesh_file.resolve()),
        "B3_DEV_mesh_node_count": node_count,
        "B3_DEV_mesh_element_count": element_count,
    }


def _dev_record_operator_contract(payload: Dict[str, Any], *, built: Dict[str, Any]) -> None:
    A_active = built["A_active"]
    M_active = built["M_active"]
    cand = built.get("cand") or {}
    act_a_fin = audit._petsc_sparse_owned_row_value_audit(A_active)
    act_m_fin = audit._petsc_sparse_owned_row_value_audit(M_active)
    a_rn = audit._petsc_sparse_owned_row_norms(A_active)
    m_rn = audit._petsc_sparse_owned_row_norms(M_active)
    a_cn = audit._petsc_sparse_owned_col_norms(A_active)
    payload["B3_DEV_structural_inactive_removed_count"] = int(cand.get("inactive_structural_count", 0))
    payload["B3_DEV_structural_inactive_origin_explained_pass"] = bool(
        audit._b3_struct_active_candidate_origin_policy_pass(cand, policy="mesh_independent")
    )
    payload["B3_DEV_Aup_supported_rows_preserved_pass"] = bool(
        int(cand.get("inactive_aup_overlap_count", 1)) == 0
    )
    payload["B3_DEV_operator_nonzero_contract_pass"] = bool(
        audit._b3_loc_nonzero_contract_pass(
            audit._mat_norm_or_none(A_active), int(audit._petsc_mat_global_nnz_used(A_active))
        )
        and audit._b3_loc_nonzero_contract_pass(
            audit._mat_norm_or_none(M_active), int(audit._petsc_mat_global_nnz_used(M_active))
        )
    )
    payload["B3_DEV_operator_contract_pass"] = bool(
        payload["B3_DEV_operator_nonzero_contract_pass"]
        and act_a_fin["all_values_finite_pass"]
        and act_m_fin["all_values_finite_pass"]
    )
    payload["B3_DEV_active_dimension"] = int(built["active_local"].size)
    payload["B3_DEV_A_shape"] = audit._mat_shape(A_active)
    payload["B3_DEV_M_shape"] = audit._mat_shape(M_active)
    payload["B3_DEV_A_all_values_finite_pass"] = bool(act_a_fin["all_values_finite_pass"])
    payload["B3_DEV_M_all_values_finite_pass"] = bool(act_m_fin["all_values_finite_pass"])
    payload["B3_DEV_A_exact_zero_row_count"] = int(np.sum(a_rn == 0.0))
    payload["B3_DEV_M_exact_zero_row_count"] = int(np.sum(m_rn == 0.0))
    payload["B3_DEV_A_exact_zero_column_count"] = int(np.sum(a_cn == 0.0))
    payload["B3_DEV_zero_row_column_cleanup_contract_pass"] = bool(
        payload["B3_DEV_A_exact_zero_row_count"] == 0
        and payload["B3_DEV_M_exact_zero_row_count"] == 0
        and payload["B3_DEV_A_exact_zero_column_count"] == 0
        and payload["B3_DEV_operator_contract_pass"]
        and payload["B3_DEV_structural_inactive_origin_explained_pass"]
        and payload["B3_DEV_Aup_supported_rows_preserved_pass"]
    )
    payload["B3_DEV_dirichlet_rows_removed"] = int(built.get("bc_rows", np.asarray([])).size)


def _dev_extract_modes_ciss(
    eps: Any,
    A_active: Any,
    built: Dict[str, Any],
    *,
    target_hz: float,
    freq_lo: float,
    freq_hi: float,
) -> Tuple[int, bool]:
    from slepc4py import SLEPc

    nconv = int(eps.getConverged())
    accepted_any = False
    free_rows = np.asarray(built["free_rows"], dtype=np.int32).ravel()
    bc_rows = np.unique(np.asarray(built["bc_rows"], dtype=np.int32).ravel())
    active_local = np.asarray(built["active_local"], dtype=np.int32).ravel()
    inactive_local = np.asarray(built["inactive_local"], dtype=np.int32).ravel()
    u_idx = np.asarray(built["u_idx"], dtype=np.int32).ravel()
    p_idx = np.asarray(built["p_idx"], dtype=np.int32).ravel()
    n_w = int(built["n_w"])
    n_free = int(free_rows.size)
    payload["B3_DEV_converged_mode_count"] = nconv

    for i in range(nconv):
        vr = A_active.createVecRight()
        vi = A_active.createVecRight()
        try:
            lam = eps.getEigenpair(i, vr, vi)
            lam_re = float(np.real(complex(lam)))
            lam_im = float(np.imag(complex(lam)))
            finite = bool(math.isfinite(lam_re) and math.isfinite(lam_im))
            f_hz = None
            if finite and abs(lam_im) <= 1.0e-12 and lam_re > 0.0:
                f_hz = math.sqrt(max(lam_re, 0.0)) / (2.0 * math.pi)
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
                audit._b3_lambda_near_unity_signature(f_hz)
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
            accepted_any = accepted_any or mode_pass
            payload[f"B3_DEV_mode_{i}_solver"] = payload.get("B3_DEV_solver_name")
            payload[f"B3_DEV_mode_{i}_lambda_real"] = _safe_float(lam_re)
            payload[f"B3_DEV_mode_{i}_lambda_imag"] = _safe_float(lam_im)
            payload[f"B3_DEV_mode_{i}_eigenvalue_finite_pass"] = finite
            payload[f"B3_DEV_mode_{i}_frequency_hz_if_real_positive"] = _safe_float(f_hz)
            payload[f"B3_DEV_mode_{i}_inside_requested_interval_pass"] = inside
            payload[f"B3_DEV_mode_{i}_target_reference_distance_hz"] = (
                _safe_float(abs(float(f_hz) - target_hz)) if f_hz is not None else None
            )
            payload[f"B3_DEV_mode_{i}_eps_compute_error_relative"] = _safe_float(eps_err)
            payload[f"B3_DEV_mode_{i}_eps_relative_error_acceptance_pass"] = eps_ok
            payload[f"B3_DEV_mode_{i}_full_vector_reconstructed"] = True
            payload[f"B3_DEV_mode_{i}_structural_inactive_zero_pass"] = si_pass
            payload[f"B3_DEV_mode_{i}_dirichlet_zero_pass"] = d_pass
            payload[f"B3_DEV_mode_{i}_u_norm"] = _safe_float(u_norm)
            payload[f"B3_DEV_mode_{i}_p_norm"] = _safe_float(p_norm)
            payload[f"B3_DEV_mode_{i}_pressure_support_metric"] = _safe_float(p_support)
            payload[f"B3_DEV_mode_{i}_lambda_one_pollution_signature"] = lambda_one
            payload[f"B3_DEV_mode_{i}_nonfinite_eigenpair_signature"] = nonfinite
            payload[f"B3_DEV_mode_{i}_acceptance_pass"] = mode_pass
        finally:
            vr.destroy()
            vi.destroy()
    payload["B3_DEV_accepted_mode_count"] = int(
        sum(1 for i in range(nconv) if payload.get(f"B3_DEV_mode_{i}_acceptance_pass"))
    )
    return nconv, accepted_any


def _run_dev_coarse_contract(pre: Dict[str, Any], mesh_variant: str) -> int:
    payload: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "B3_dev_coarse_corrected_operator_contract_only",
        **_dev_mesh_bootstrap_payload(mesh_variant),
        "B3_DEV_operator_contract_pass": False,
        "no_new_eigensolve_executed": True,
        "production_promotion": "BLOCKED",
    }
    timer = _B3DevTiming(payload)
    timer.mark("total_begin")
    mats: List[Any] = []
    seen: set[int] = set()
    verdict = "B3_DEV_COARSE_OPERATOR_CONTRACT_BLOCKED"
    try:
        if not pre.get("preassembly_contract_pass"):
            return 2
        if MPI.COMM_WORLD.size != 1:
            return 2
        mesh_path_file = Path(payload["B3_DEV_mesh_path"])
        if not mesh_path_file.is_file():
            payload["B3_DEV_failure_reason"] = f"mesh_missing:{mesh_path_file}"
            return 2
        timer.mark("operator_build_begin")
        built = audit._b3_build_corrected_structural_active_operators(
            mats_to_destroy=mats,
            mat_destroy_seen=seen,
            mesh_level=mesh_variant,
            struct_active_count_policy="mesh_independent",
        )
        timer.mark("operator_build_end")
        payload["B3_DEV_operator_build_elapsed_seconds"] = payload.get(
            "B3_DEV_timing_operator_build_end_elapsed_seconds"
        )
        _dev_record_operator_contract(payload, built=built)
        if payload["B3_DEV_zero_row_column_cleanup_contract_pass"]:
            verdict = "B3_DEV_COARSE_OPERATOR_CONTRACT_PASS"
            rc = 0
        else:
            rc = 2
    except audit._B3StructActiveBuildError as exc:
        payload["B3_DEV_failure_stage"] = exc.stage
        payload["B3_DEV_failure_reason"] = exc.reason
        rc = 2
    except Exception as exc:
        payload["B3_DEV_failure_reason"] = f"{type(exc).__name__}:{exc}"
        rc = 2
    finally:
        timer.mark("total_end")
        timer.finalize()
        payload["next_step_verdict"] = verdict
        audit._write_json_atomic(OUT_JSON_DEV_CONTRACT, payload)
        audit._destroy_mats_deduped(mats)
    return rc


def _dev_record_ciss_direct_stable_mirror(payload: Dict[str, Any]) -> None:
    """Copy shared direct-stable introspection into B3_DEV_CISS_* report fields."""
    payload["B3_DEV_CISS_ST_type_effective"] = payload.get("B3_CISS_direct_stable_ST_type_effective")
    payload["B3_DEV_CISS_KSP_type_effective"] = payload.get("B3_CISS_direct_stable_KSP_type_effective")
    payload["B3_DEV_CISS_PC_type_effective"] = payload.get("B3_CISS_direct_stable_PC_type_effective")
    payload["B3_DEV_CISS_factor_solver_effective"] = payload.get("B3_CISS_direct_stable_factor_solver_effective")
    payload["B3_DEV_CISS_factor_shift_verification_classification"] = payload.get(
        "B3_CISS_direct_stable_factor_shift_verification_classification"
    )


def _run_dev_coarse_ciss_benchmark(pre: Dict[str, Any], mesh_variant: str) -> int:
    from slepc4py import SLEPc

    lam_lo = audit._b3_hz_to_lambda_sq(audit.B3_CISS_VALIDATION_FREQ_LO_HZ)
    lam_hi = audit._b3_hz_to_lambda_sq(audit.B3_CISS_VALIDATION_FREQ_HI_HZ)
    target_hz = float(audit.B3_CISS_VALIDATION_TARGET_HZ)
    payload: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "B3_dev_coarse_ciss_direct_stable_benchmark_only",
        "B3_DEV_solver_name": "CISS-DIRECT-STABLE",
        **_dev_mesh_bootstrap_payload(mesh_variant),
        "B3_DEV_validation_frequency_interval_hz": [audit.B3_CISS_VALIDATION_FREQ_LO_HZ, audit.B3_CISS_VALIDATION_FREQ_HI_HZ],
        "B3_DEV_validation_lambda_interval": [_safe_float(lam_lo), _safe_float(lam_hi)],
        "B3_DEV_execution_authorized": True,
        "B3_DEV_execution_fallback_used": False,
        "B3_DEV_execution_automatic_retry_used": False,
        "B3_DEV_execution_additional_EPS_solve_used": False,
        "no_new_eigensolve_executed": True,
        "new_eigensolve_executed": False,
        "production_promotion": "BLOCKED",
    }
    timer = _B3DevTiming(payload)
    mats: List[Any] = []
    seen: set[int] = set()
    eps = None
    verdict = "B3_DEV_COARSE_CISS_BENCHMARK_BLOCKED_BY_SOLVER_INTERFACE"
    try:
        if not pre.get("preassembly_contract_pass") or MPI.COMM_WORLD.size != 1:
            return 2
        if not Path(payload["B3_DEV_mesh_path"]).is_file():
            payload["B3_DEV_failure_reason"] = "mesh_missing"
            return 2
        timer.mark("operator_build_begin")
        built = audit._b3_build_corrected_structural_active_operators(
            mats_to_destroy=mats,
            mat_destroy_seen=seen,
            mesh_level=mesh_variant,
            struct_active_count_policy="mesh_independent",
        )
        timer.mark("A_active_M_active_ready")
        _dev_record_operator_contract(payload, built=built)
        if not payload["B3_DEV_zero_row_column_cleanup_contract_pass"]:
            payload["B3_DEV_failure_reason"] = "operator_contract_failed"
            return 2

        timer.mark("eps_configure_begin")
        ciss_type = getattr(SLEPc.EPS.Type, "CISS", None)
        if ciss_type is None:
            payload["B3_DEV_failure_reason"] = "CISS_unavailable"
            return 2
        eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
        payload["B3_DEV_CISS_EPS_created"] = True
        payload["B3_DEV_CISS_setup_calls_setup"] = False
        payload["B3_DEV_CISS_solve_attempted"] = False
        payload["B3_DEV_CISS_solve_count"] = 0
        payload["B3_DEV_CISS_converged_mode_count"] = 0
        A_active = built["A_active"]
        M_active = built["M_active"]
        eps.setOperators(A_active, M_active)
        eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        eps.setType(ciss_type)
        audit._b3_ciss_configure_rg_interval(eps, lam_lo=lam_lo, lam_hi=lam_hi)
        audit._b3_ciss_apply_optional_sizes(eps, payload, n_active=int(A_active.getSize()[0]))
        ok, reason = audit._b3_ciss_apply_direct_stable_st_ksp_pc_policy(eps, payload)
        if not ok:
            payload["B3_DEV_failure_reason"] = reason
            return 2
        timer.mark("eps_configure_end")

        timer.mark("eps_setup_begin")
        eps.setUp()
        payload["B3_DEV_CISS_setup_calls_setup"] = True
        timer.mark("eps_setup_end")
        payload.update(audit._b3_ciss_introspect_direct_stable_after_setup(eps))
        audit._b3_ciss_finalize_direct_stable_factor_shift_verification(eps, payload)
        _dev_record_ciss_direct_stable_mirror(payload)
        payload["B3_DEV_CISS_setup_elapsed_seconds"] = _safe_float(
            float(payload.get("B3_DEV_timing_eps_setup_end_elapsed_seconds", 0))
            - float(payload.get("B3_DEV_timing_eps_setup_begin_elapsed_seconds", 0))
        )
        if not audit._b3_ciss_direct_stable_policy_effective_pass(payload):
            payload["B3_DEV_failure_reason"] = "direct_stable_setup_not_verified"
            return 2

        timer.mark("eps_solve_begin")
        payload["B3_DEV_CISS_solve_attempted"] = True
        eps.solve()
        payload["B3_DEV_CISS_solve_count"] = 1
        timer.mark("eps_solve_end")
        payload["new_eigensolve_executed"] = True
        payload["no_new_eigensolve_executed"] = False

        nconv, accepted = _dev_extract_modes_ciss(
            eps,
            A_active,
            built,
            target_hz=target_hz,
            freq_lo=float(audit.B3_CISS_VALIDATION_FREQ_LO_HZ),
            freq_hi=float(audit.B3_CISS_VALIDATION_FREQ_HI_HZ),
        )
        payload["B3_DEV_CISS_converged_mode_count"] = int(nconv)
        payload["B3_DEV_setup_elapsed_seconds"] = payload.get("B3_DEV_CISS_setup_elapsed_seconds")
        payload["B3_DEV_solve_elapsed_seconds"] = _safe_float(
            float(payload.get("B3_DEV_timing_eps_solve_end_elapsed_seconds", 0))
            - float(payload.get("B3_DEV_timing_eps_solve_begin_elapsed_seconds", 0))
        )
        payload["B3_DEV_CISS_solve_elapsed_seconds"] = payload["B3_DEV_solve_elapsed_seconds"]
        if accepted:
            verdict = "B3_DEV_COARSE_CISS_BENCHMARK_PASS"
            rc = 0
        elif nconv > 0:
            verdict = "B3_DEV_COARSE_CISS_BENCHMARK_COMPLETED_NO_ACCEPTABLE_MODE"
            rc = 2
        else:
            verdict = "B3_DEV_COARSE_CISS_BENCHMARK_COMPLETED_ZERO_MODES"
            rc = 2
    except Exception as exc:
        payload["B3_DEV_failure_reason"] = f"{type(exc).__name__}:{exc}"
        rc = 2
    finally:
        if eps is not None:
            try:
                eps.destroy()
            except Exception:
                pass
        timer.finalize()
        payload["B3_DEV_CISS_total_elapsed_seconds"] = _safe_float(
            payload.get("B3_DEV_timing_total_wall_elapsed_seconds")
        )
        payload["next_step_verdict"] = verdict
        audit._write_json_atomic(OUT_JSON_DEV_CISS, payload)
        audit._destroy_mats_deduped(mats)
    return rc


def _run_dev_coarse_st_benchmark(pre: Dict[str, Any], mesh_variant: str) -> int:
    from slepc4py import SLEPc

    target_hz = float(audit.B3_CISS_VALIDATION_TARGET_HZ)
    target_lambda = audit._b3_hz_to_lambda_sq(target_hz)
    lam_lo = audit._b3_hz_to_lambda_sq(audit.B3_CISS_VALIDATION_FREQ_LO_HZ)
    lam_hi = audit._b3_hz_to_lambda_sq(audit.B3_CISS_VALIDATION_FREQ_HI_HZ)
    payload: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "B3_dev_coarse_krylovschur_sinvert_benchmark_only",
        "B3_DEV_solver_name": "KRYLOVSCHUR-ST-SINVERT-MUMPS",
        **_dev_mesh_bootstrap_payload(mesh_variant),
        "B3_DEV_validation_frequency_interval_hz": [audit.B3_CISS_VALIDATION_FREQ_LO_HZ, audit.B3_CISS_VALIDATION_FREQ_HI_HZ],
        "B3_DEV_st_shift_lambda": _safe_float(target_lambda),
        "B3_DEV_execution_authorized": True,
        "B3_DEV_execution_fallback_used": False,
        "B3_DEV_execution_automatic_retry_used": False,
        "B3_DEV_execution_additional_EPS_solve_used": False,
        "no_new_eigensolve_executed": True,
        "new_eigensolve_executed": False,
        "production_promotion": "BLOCKED",
    }
    timer = _B3DevTiming(payload)
    mats: List[Any] = []
    seen: set[int] = set()
    eps = None
    verdict = "B3_DEV_COARSE_ST_BENCHMARK_BLOCKED_BY_SOLVER_INTERFACE"
    try:
        if not pre.get("preassembly_contract_pass") or MPI.COMM_WORLD.size != 1:
            return 2
        timer.mark("operator_build_begin")
        built = audit._b3_build_corrected_structural_active_operators(
            mats_to_destroy=mats,
            mat_destroy_seen=seen,
            mesh_level=mesh_variant,
            struct_active_count_policy="mesh_independent",
        )
        timer.mark("A_active_M_active_ready")
        _dev_record_operator_contract(payload, built=built)
        if not payload["B3_DEV_zero_row_column_cleanup_contract_pass"]:
            payload["B3_DEV_failure_reason"] = "operator_contract_failed"
            return 2

        timer.mark("eps_configure_begin")
        eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
        A_active = built["A_active"]
        M_active = built["M_active"]
        eps.setOperators(A_active, M_active)
        eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        try:
            eps.setDimensions(nev=B3_DEV_COARSE_NEV, ncv=B3_DEV_COARSE_NCV)
        except TypeError:
            eps.setDimensions(B3_DEV_COARSE_NEV, B3_DEV_COARSE_NCV)
        st = eps.getST()
        try:
            st.setType(SLEPc.ST.Type.SINVERT)
        except Exception:
            st.setType("sinvert")
        st.setShift(float(target_lambda))
        ksp = st.getKSP()
        pc = ksp.getPC()
        ok, reason = audit._b3_ciss_require_mumps_factor_solver(pc)
        if not ok:
            payload["B3_DEV_failure_reason"] = reason
            return 2
        import fem_main_3d as fem3d

        fem3d._slepc_configure_st_ksp_pc(
            ksp,
            pc,
            audit._b3_ciss_direct_stable_solver_cfg(),
            block_is=None,
            opts_prefix="st_",
            use_ciss=True,
        )
        audit._b3_ciss_record_direct_stable_factor_shift_request(pc, payload)
        audit._b3_ciss_apply_st_mumps_icntl_petsc_options()
        timer.mark("eps_configure_end")

        timer.mark("eps_setup_begin")
        eps.setUp()
        timer.mark("eps_setup_end")

        timer.mark("eps_solve_begin")
        eps.solve()
        timer.mark("eps_solve_end")
        payload["new_eigensolve_executed"] = True
        payload["no_new_eigensolve_executed"] = False

        nconv, accepted = _dev_extract_modes_ciss(
            eps,
            A_active,
            built,
            target_hz=target_hz,
            freq_lo=float(audit.B3_CISS_VALIDATION_FREQ_LO_HZ),
            freq_hi=float(audit.B3_CISS_VALIDATION_FREQ_HI_HZ),
        )
        payload["B3_DEV_setup_elapsed_seconds"] = _safe_float(
            float(payload.get("B3_DEV_timing_eps_setup_end_elapsed_seconds", 0))
            - float(payload.get("B3_DEV_timing_eps_setup_begin_elapsed_seconds", 0))
        )
        payload["B3_DEV_solve_elapsed_seconds"] = _safe_float(
            float(payload.get("B3_DEV_timing_eps_solve_end_elapsed_seconds", 0))
            - float(payload.get("B3_DEV_timing_eps_solve_begin_elapsed_seconds", 0))
        )
        if accepted:
            verdict = "B3_DEV_COARSE_ST_BENCHMARK_PASS"
            rc = 0
        elif nconv > 0:
            verdict = "B3_DEV_COARSE_ST_BENCHMARK_COMPLETED_NO_ACCEPTABLE_MODE"
            rc = 2
        else:
            verdict = "B3_DEV_COARSE_ST_BENCHMARK_COMPLETED_ZERO_MODES"
            rc = 2
    except Exception as exc:
        payload["B3_DEV_failure_reason"] = f"{type(exc).__name__}:{exc}"
        rc = 2
    finally:
        if eps is not None:
            try:
                eps.destroy()
            except Exception:
                pass
        timer.finalize()
        payload["next_step_verdict"] = verdict
        audit._write_json_atomic(OUT_JSON_DEV_ST, payload)
        audit._destroy_mats_deduped(mats)
    return rc


def _run_dev_compare_summary(mesh_variant: str) -> int:
    rows: List[Dict[str, Any]] = []

    def _load_row(path: Path, solver: str) -> None:
        if not path.is_file():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        freqs: List[float] = []
        best_dist = None
        target_hz = float(audit.B3_CISS_VALIDATION_TARGET_HZ)
        nconv = int(data.get("B3_DEV_converged_mode_count") or 0)
        for i in range(nconv):
            f = data.get(f"B3_DEV_mode_{i}_frequency_hz_if_real_positive")
            if f is not None and math.isfinite(float(f)):
                freqs.append(float(f))
                d = abs(float(f) - target_hz)
                if best_dist is None or d < best_dist:
                    best_dist = d
        rows.append(
            {
                "solver": solver,
                "mesh_variant": data.get("B3_DEV_mesh_variant", mesh_variant),
                "active_dimension": data.get("B3_DEV_active_dimension"),
                "operator_build_time_s": data.get("B3_DEV_operator_build_elapsed_seconds")
                or data.get("B3_DEV_timing_operator_build_end_elapsed_seconds"),
                "setup_time_s": data.get("B3_DEV_setup_elapsed_seconds"),
                "solve_time_s": data.get("B3_DEV_solve_elapsed_seconds"),
                "total_time_s": data.get("B3_DEV_timing_total_wall_elapsed_seconds"),
                "returned_frequencies_hz": freqs,
                "closest_to_244_39_hz": _safe_float(
                    (target_hz - best_dist) if best_dist is not None else None
                ),
                "closest_distance_hz": _safe_float(best_dist),
                "accepted_mode_count": data.get("B3_DEV_accepted_mode_count"),
                "peak_memory_rss_mb": data.get("B3_DEV_peak_memory_rss_mb"),
                "verdict": data.get("next_step_verdict"),
            }
        )

    _load_row(OUT_JSON_DEV_CONTRACT, "operator_contract")
    _load_row(OUT_JSON_DEV_CISS, "CISS-DIRECT-STABLE")
    _load_row(OUT_JSON_DEV_ST, "KRYLOVSCHUR-ST-SINVERT-MUMPS")

    summary = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "B3_dev_solver_benchmark_compare_only",
        "B3_DEV_mesh_variant": mesh_variant,
        "B3_DEV_mesh_is_solver_smoke_test_only": True,
        "B3_DEV_mesh_not_authorized_for_final_physics_validation": True,
        "comparison_rows": rows,
        "no_automatic_production_winner_selected": True,
    }
    audit._write_json_atomic(OUT_JSON_DEV_SUMMARY, summary)
    OUT_MD_DEV_SUMMARY.write_text(
        "# B3 dev coarse solver benchmark summary\n\n"
        + "\n".join(
            f"- **{r['solver']}**: active_dim={r.get('active_dimension')} "
            f"build={r.get('operator_build_time_s')}s setup={r.get('setup_time_s')}s "
            f"solve={r.get('solve_time_s')}s total={r.get('total_time_s')}s "
            f"accepted={r.get('accepted_mode_count')} verdict={r.get('verdict')}"
            for r in rows
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[B3_DEV] summary_written={OUT_JSON_DEV_SUMMARY}", flush=True)
    return 0


def is_b3_dev_mode(argv: List[str]) -> bool:
    dev_flags = (
        B3_DEV_COARSE_CONTRACT_ARG,
        B3_DEV_COARSE_CISS_ARG,
        B3_DEV_COARSE_ST_ARG,
        B3_DEV_COMPARE_ARG,
    )
    return any(f in argv for f in dev_flags)


def run_b3_dev_mode(argv: List[str], pre: Dict[str, Any]) -> int:
    mesh_variant = _parse_dev_mesh_variant(argv)
    if mesh_variant != B3_DEV_DEFAULT_MESH_VARIANT:
        print(f"[B3_DEV] unsupported mesh_variant={mesh_variant}", flush=True)
        return 2
    if B3_DEV_COARSE_CONTRACT_ARG in argv:
        return _run_dev_coarse_contract(pre, mesh_variant)
    if B3_DEV_COARSE_CISS_ARG in argv:
        return _run_dev_coarse_ciss_benchmark(pre, mesh_variant)
    if B3_DEV_COARSE_ST_ARG in argv:
        return _run_dev_coarse_st_benchmark(pre, mesh_variant)
    if B3_DEV_COMPARE_ARG in argv:
        return _run_dev_compare_summary(mesh_variant)
    print("[B3_DEV] no dev mode flag recognized", flush=True)
    return 2
