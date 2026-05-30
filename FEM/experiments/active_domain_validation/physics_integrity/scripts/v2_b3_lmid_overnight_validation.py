#!/usr/bin/env python3
"""Overnight L_mid CISS reference + ST multi-target validation (single shared operator build)."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import run_v2_B3_trace_coupled_operator_and_seed_transfer_audit as audit
import v2_b3_dev_solver_benchmark as dev_bench

B3_LMID_OVERNIGHT_ARG = "--B3-Lmid-overnight-CISS-ST-multi-target-validation-only"
B3_LMID_CISS_ONLY_ARG = "--B3-Lmid-overnight-CISS-reference-only"
B3_LMID_COMPARE_ONLY_ARG = "--B3-Lmid-overnight-ST-vs-CISS-comparison-only"
B3_LMID_MODE_ARGS = (
    B3_LMID_OVERNIGHT_ARG,
    B3_LMID_CISS_ONLY_ARG,
    B3_LMID_COMPARE_ONLY_ARG,
)
LMID_MESH_LEVEL = "L_mid"

LMID_ST_TARGETS_HZ = [221.5, 227.0, 232.5, 238.0, 243.5, 249.0, 254.5, 260.0, 264.0]

OUT_JSON_LMID_CISS = audit.CONV_DIAG / "v2_B3_Lmid_overnight_CISS_reference_only.json"
OUT_MD_LMID_CISS = audit.CONV_DIAG / "v2_B3_Lmid_overnight_CISS_reference_only.md"
OUT_JSON_LMID_ST = audit.CONV_DIAG / "v2_B3_Lmid_overnight_ST_multi_target_only.json"
OUT_MD_LMID_ST = audit.CONV_DIAG / "v2_B3_Lmid_overnight_ST_multi_target_only.md"
OUT_JSON_LMID_SUMMARY = audit.CONV_DIAG / "v2_B3_Lmid_overnight_CISS_ST_multi_target_validation_summary.json"
OUT_MD_LMID_SUMMARY = audit.CONV_DIAG / "v2_B3_Lmid_overnight_CISS_ST_multi_target_validation_summary.md"
OUT_JSON_LMID_CONTRACT_CKPT = audit.CONV_DIAG / "v2_B3_Lmid_overnight_operator_contract_checkpoint.json"
OUT_JSON_LMID_ST_CISS_CKPT = audit.CONV_DIAG / "v2_B3_Lmid_overnight_ST_vs_CISS_comparison_checkpoint.json"
OUT_JSON_LMID_CROSS_MESH_CKPT = audit.CONV_DIAG / "v2_B3_Lmid_overnight_cross_mesh_comparison_checkpoint.json"

DENSE_MESH_VARIANT = "L_dev_dense"
DENSE_ST_TARGETS_HZ = [224.0, 234.0, 244.39, 255.0, 262.5]
CROSS_MESH_DRIFT_TOL_HZ = 1.0
CROSS_MESH_DRIFT_TOL_REL = 0.005


def is_lmid_overnight_mode(argv: List[str]) -> bool:
    return any(arg in argv for arg in B3_LMID_MODE_ARGS)


def is_lmid_ciss_only_mode(argv: List[str]) -> bool:
    return B3_LMID_CISS_ONLY_ARG in argv


def is_lmid_st_ciss_compare_only_mode(argv: List[str]) -> bool:
    return B3_LMID_COMPARE_ONLY_ARG in argv


def _lmid_mesh_path() -> Path:
    return audit.mesh_path(LMID_MESH_LEVEL, audit.CASE_ID)


def _write_report(path_json: Path, path_md: Path, payload: Dict[str, Any], *, title: str) -> None:
    audit._write_json_atomic(path_json, payload)
    path_md.write_text(
        f"# {title}\n\n"
        f"- verdict: `{payload.get('next_step_verdict')}`\n"
        f"- failure_reason: {payload.get('B3_Lmid_failure_reason')}\n",
        encoding="utf-8",
    )


def _write_checkpoint(path_json: Path, payload: Dict[str, Any]) -> None:
    payload["checkpoint_written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    audit._write_json_atomic(path_json, payload)


def _per_mode_diagnostics_from_payload(
    data: Dict[str, Any], *, nconv: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Scalar diagnostics already produced by shared mode extraction (no new physics)."""
    if nconv is None:
        nconv = int(data.get("B3_Lmid_CISS_converged_mode_count") or data.get("B3_DEV_converged_mode_count") or 0)
    rows: List[Dict[str, Any]] = []
    for i in range(nconv):
        rows.append(
            {
                "mode_index": i,
                "frequency_hz": data.get(f"B3_DEV_mode_{i}_frequency_hz_if_real_positive"),
                "eps_compute_error_relative": data.get(f"B3_DEV_mode_{i}_eps_compute_error_relative"),
                "eps_relative_error_acceptance_pass": data.get(f"B3_DEV_mode_{i}_eps_relative_error_acceptance_pass"),
                "acceptance_pass": data.get(f"B3_DEV_mode_{i}_acceptance_pass"),
                "pressure_support_metric": data.get(f"B3_DEV_mode_{i}_pressure_support_metric"),
                "u_norm": data.get(f"B3_DEV_mode_{i}_u_norm"),
                "p_norm": data.get(f"B3_DEV_mode_{i}_p_norm"),
                "structural_inactive_zero_pass": data.get(f"B3_DEV_mode_{i}_structural_inactive_zero_pass"),
                "dirichlet_zero_pass": data.get(f"B3_DEV_mode_{i}_dirichlet_zero_pass"),
            }
        )
    return rows


def _modal_facet_proxy_export_note() -> Dict[str, Any]:
    """Facet-tag proxies need parent-mesh DOF maps; not available from built operator path alone."""
    return {
        "B3_Lmid_modal_observables_export_scope": "INFORMATIONAL_ONLY",
        "B3_Lmid_modal_observables_already_in_per_mode_fields": [
            "pressure_support_metric",
            "u_norm",
            "p_norm",
        ],
        "B3_Lmid_modal_facet_proxies_requested": [
            "top_plate_motion_proxy",
            "soundhole_flow_proxy",
            "internal_pressure_probe_proxy",
        ],
        "B3_Lmid_modal_facet_proxies_export_pass": False,
        "B3_Lmid_modal_facet_proxies_export_reason": (
            "facet_tag_dof_mapping_requires_parent_mesh_reload;"
            "not_in_shared_operator_built_path;deferred_to_avoid_overnight_risk"
        ),
        "B3_Lmid_modal_facet_proxies_do_not_affect_solver_acceptance": True,
    }


def _lmid_record_ciss_direct_stable_mirror(payload: Dict[str, Any]) -> None:
    """Copy shared direct-stable introspection into B3_Lmid_CISS_* report fields."""
    payload["B3_Lmid_CISS_ST_type_effective"] = payload.get("B3_CISS_direct_stable_ST_type_effective")
    payload["B3_Lmid_CISS_KSP_type_effective"] = payload.get("B3_CISS_direct_stable_KSP_type_effective")
    payload["B3_Lmid_CISS_PC_type_effective"] = payload.get("B3_CISS_direct_stable_PC_type_effective")
    payload["B3_Lmid_CISS_factor_solver_effective"] = payload.get("B3_CISS_direct_stable_factor_solver_effective")
    payload["B3_Lmid_CISS_factor_shift_verification_classification"] = payload.get(
        "B3_CISS_direct_stable_factor_shift_verification_classification"
    )
    payload["B3_Lmid_CISS_direct_stable_setup_verified_pass"] = audit._b3_ciss_direct_stable_policy_effective_pass(
        payload
    )


def _st_deduplicated_provenance(
    accepted_modes: List[Dict[str, Any]],
    union_freqs: List[float],
    *,
    tol_hz: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for f_u in union_freqs:
        contributors: List[Dict[str, Any]] = []
        for m in accepted_modes:
            f_m = float(m.get("frequency_hz", float("nan")))
            if not math.isfinite(f_m):
                continue
            if abs(f_m - float(f_u)) <= tol_hz:
                contributors.append(
                    {
                        "st_shift_target_hz": m.get("st_shift_target_hz"),
                        "mode_index_at_target": m.get("mode_index"),
                        "frequency_hz": f_m,
                        "eps_compute_error_relative": m.get("eps_compute_error_relative"),
                    }
                )
        out.append(
            {
                "frequency_hz": float(f_u),
                "contributing_st_shift_targets_hz": sorted(
                    {float(c["st_shift_target_hz"]) for c in contributors if c.get("st_shift_target_hz") is not None}
                ),
                "contributor_count": len(contributors),
                "contributors": contributors,
            }
        )
    return out


def _accepted_frequencies_from_mode_payload(data: Dict[str, Any], *, prefix: str = "B3_DEV") -> List[float]:
    freqs: List[float] = []
    nconv = int(data.get(f"{prefix}_converged_mode_count") or data.get("B3_DEV_converged_mode_count") or 0)
    for i in range(nconv):
        if not data.get(f"{prefix}_mode_{i}_acceptance_pass", data.get(f"B3_DEV_mode_{i}_acceptance_pass")):
            continue
        f = data.get(f"{prefix}_mode_{i}_frequency_hz_if_real_positive")
        if f is not None and math.isfinite(float(f)):
            freqs.append(float(f))
    return sorted(freqs)


def _lmid_operator_contract_pass(payload: Dict[str, Any], *, built: Dict[str, Any]) -> bool:
    dev_bench._dev_record_operator_contract(payload, built=built)
    active_dim = int(built["active_local"].size)
    expected = int(audit.B3_STRUCT_ACTIVE_ACTIVE_DIM_EXPECTED)
    payload["B3_Lmid_active_dimension"] = active_dim
    payload["B3_Lmid_active_dimension_expected"] = expected
    payload["B3_Lmid_active_dimension_contract_pass"] = bool(active_dim == expected)
    payload["B3_Lmid_A_shape"] = audit._mat_shape(built["A_active"])
    payload["B3_Lmid_M_shape"] = audit._mat_shape(built["M_active"])
    payload["B3_Lmid_operator_contract_pass"] = bool(
        payload.get("B3_DEV_operator_contract_pass")
        and payload.get("B3_DEV_zero_row_column_cleanup_contract_pass")
        and payload["B3_Lmid_active_dimension_contract_pass"]
    )
    return bool(payload["B3_Lmid_operator_contract_pass"])


def _greedy_frequency_pairs(
    ref_freqs: List[float],
    test_freqs: List[float],
    *,
    match_tol_hz: float,
) -> Tuple[List[Tuple[float, float, float]], List[float], List[float]]:
    """Pairs (ref, test, abs_diff_hz); unmatched ref and test lists."""
    pool = list(test_freqs)
    pairs: List[Tuple[float, float, float]] = []
    missing: List[float] = []
    for f_ref in ref_freqs:
        best_j = None
        best_d = None
        for j, f_t in enumerate(pool):
            d = abs(f_t - f_ref)
            if d <= match_tol_hz and (best_d is None or d < best_d):
                best_d = d
                best_j = j
        if best_j is None:
            missing.append(float(f_ref))
        else:
            f_t = pool.pop(best_j)
            pairs.append((float(f_ref), float(f_t), float(best_d or 0.0)))
    return pairs, missing, pool


def _cross_mesh_convergence_report(
    *,
    lmid_freqs: List[float],
    dense_ciss_freqs: List[float],
    dense_st_freqs: List[float],
) -> Dict[str, Any]:
    pairs_ciss, miss_ciss, extra_ciss = _greedy_frequency_pairs(
        dense_ciss_freqs, lmid_freqs, match_tol_hz=CROSS_MESH_DRIFT_TOL_HZ
    )
    pairs_st, miss_st, extra_st = _greedy_frequency_pairs(
        dense_st_freqs, lmid_freqs, match_tol_hz=CROSS_MESH_DRIFT_TOL_HZ
    )
    drifts_hz = [d for _, _, d in pairs_ciss]
    rel_drifts = [
        (d / max(abs(r), 1.0e-30)) for (r, _, d) in pairs_ciss
    ]
    max_d = max(drifts_hz) if drifts_hz else None
    max_rel = max(rel_drifts) if rel_drifts else None
    same_count = bool(len(lmid_freqs) == len(dense_ciss_freqs) == len(dense_st_freqs))
    drift_ok = bool(
        max_d is not None
        and max_rel is not None
        and max_d <= CROSS_MESH_DRIFT_TOL_HZ
        and max_rel <= CROSS_MESH_DRIFT_TOL_REL
    )
    return {
        "B3_Lmid_cross_mesh_dense_active_dimension": 41501,
        "B3_Lmid_cross_mesh_dense_CISS_accepted_mode_count": len(dense_ciss_freqs),
        "B3_Lmid_cross_mesh_dense_ST_accepted_mode_count": len(dense_st_freqs),
        "B3_Lmid_cross_mesh_Lmid_accepted_mode_count": len(lmid_freqs),
        "B3_Lmid_cross_mesh_mode_count_difference_Lmid_minus_dense_CISS": int(
            len(lmid_freqs) - len(dense_ciss_freqs)
        ),
        "B3_Lmid_cross_mesh_matched_mode_pairs_vs_dense_CISS": [
            {"dense_hz": r, "Lmid_hz": t, "abs_drift_hz": d} for r, t, d in pairs_ciss
        ],
        "B3_Lmid_cross_mesh_unmatched_dense_CISS_frequencies": miss_ciss,
        "B3_Lmid_cross_mesh_unmatched_Lmid_frequencies_vs_dense_CISS": extra_ciss,
        "B3_Lmid_cross_mesh_maximum_absolute_drift_hz": dev_bench._safe_float(max_d),
        "B3_Lmid_cross_mesh_maximum_relative_drift_fraction": dev_bench._safe_float(max_rel),
        "B3_Lmid_cross_mesh_maximum_relative_drift_percent": dev_bench._safe_float(
            (max_rel * 100.0) if max_rel is not None else None
        ),
        "B3_Lmid_cross_mesh_same_mode_count_required_for_convergence_candidate": True,
        "B3_Lmid_cross_mesh_same_mode_count_pass": same_count,
        "B3_Lmid_cross_mesh_drift_threshold_hz": CROSS_MESH_DRIFT_TOL_HZ,
        "B3_Lmid_cross_mesh_drift_threshold_relative": CROSS_MESH_DRIFT_TOL_REL,
        "B3_Lmid_cross_mesh_drift_within_threshold_pass": drift_ok,
        "B3_Lmid_cross_mesh_modal_spectrum_stable_under_threshold_pass": bool(
            same_count and drift_ok and len(miss_ciss) == 0
        ),
        "B3_Lmid_cross_mesh_automatic_production_promotion": "BLOCKED",
    }


def _load_dense_reference_payloads() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    dense_ciss_path = dev_bench._dev_out_json(dev_bench._DEV_JSON_STEM_CISS, DENSE_MESH_VARIANT)
    dense_st_path = dev_bench._dev_out_json_st_multi_target(DENSE_MESH_VARIANT, DENSE_ST_TARGETS_HZ)
    dense_ciss: Dict[str, Any] = {}
    dense_st: Dict[str, Any] = {}
    if dense_ciss_path.is_file():
        dense_ciss = json.loads(dense_ciss_path.read_text(encoding="utf-8"))
    if dense_st_path.is_file():
        dense_st = json.loads(dense_st_path.read_text(encoding="utf-8"))
    return dense_ciss, dense_st


def _run_lmid_ciss_reference(
    *,
    built: Dict[str, Any],
) -> Tuple[Dict[str, Any], bool, List[float]]:
    from slepc4py import SLEPc

    lam_lo = audit._b3_hz_to_lambda_sq(audit.B3_CISS_VALIDATION_FREQ_LO_HZ)
    lam_hi = audit._b3_hz_to_lambda_sq(audit.B3_CISS_VALIDATION_FREQ_HI_HZ)
    target_hz = float(audit.B3_CISS_VALIDATION_TARGET_HZ)
    payload: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "B3_Lmid_overnight_CISS_reference_only",
        "B3_Lmid_mesh_level": LMID_MESH_LEVEL,
        "B3_Lmid_mesh_path": str(_lmid_mesh_path().resolve()),
        "B3_Lmid_validation_frequency_interval_hz": [
            audit.B3_CISS_VALIDATION_FREQ_LO_HZ,
            audit.B3_CISS_VALIDATION_FREQ_HI_HZ,
        ],
        "B3_Lmid_solver_name": "CISS-DIRECT-STABLE",
        "no_new_eigensolve_executed": True,
        "new_eigensolve_executed": False,
        "production_promotion": "BLOCKED",
    }
    timer_ciss = dev_bench._B3DevTiming(payload)
    eps = None
    ciss_ok = False
    ciss_freqs: List[float] = []
    verdict = "B3_Lmid_OVERNIGHT_CISS_REFERENCE_BLOCKED"
    try:
        ciss_type = getattr(SLEPc.EPS.Type, "CISS", None)
        if ciss_type is None:
            payload["B3_Lmid_failure_reason"] = "CISS_unavailable"
            return payload, False, []

        timer_ciss.mark("eps_configure_begin")
        eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
        A_active = built["A_active"]
        M_active = built["M_active"]
        eps.setOperators(A_active, M_active)
        eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        eps.setType(ciss_type)
        audit._b3_ciss_configure_rg_interval(eps, lam_lo=lam_lo, lam_hi=lam_hi)
        audit._b3_ciss_apply_optional_sizes(eps, payload, n_active=int(A_active.getSize()[0]))
        ok, reason = audit._b3_ciss_apply_direct_stable_st_ksp_pc_policy(eps, payload)
        if not ok:
            payload["B3_Lmid_failure_reason"] = reason
            return payload, False, []

        timer_ciss.mark("eps_setup_begin")
        eps.setUp()
        payload["B3_Lmid_CISS_setup_calls_setup"] = True
        payload["B3_CISS_direct_stable_setup_calls_setup"] = True
        timer_ciss.mark("eps_setup_end")
        payload.update(audit._b3_ciss_introspect_direct_stable_after_setup(eps))
        audit._b3_ciss_finalize_direct_stable_factor_shift_verification(eps, payload)
        dev_bench._dev_record_ciss_direct_stable_mirror(payload)
        _lmid_record_ciss_direct_stable_mirror(payload)
        payload["B3_Lmid_CISS_setup_elapsed_seconds"] = dev_bench._safe_float(
            float(payload.get("B3_DEV_timing_eps_setup_end_elapsed_seconds", 0))
            - float(payload.get("B3_DEV_timing_eps_setup_begin_elapsed_seconds", 0))
        )
        if not audit._b3_ciss_direct_stable_policy_effective_pass(payload):
            payload["B3_Lmid_failure_reason"] = "direct_stable_setup_not_verified"
            return payload, False, []

        timer_ciss.mark("eps_solve_begin")
        payload["B3_Lmid_CISS_solve_attempted"] = True
        eps.solve()
        payload["B3_Lmid_CISS_solve_count"] = 1
        timer_ciss.mark("eps_solve_end")
        payload["new_eigensolve_executed"] = True
        payload["no_new_eigensolve_executed"] = False

        nconv, accepted = dev_bench._dev_extract_modes_ciss(
            eps,
            A_active,
            built,
            payload,
            target_hz=target_hz,
            freq_lo=float(audit.B3_CISS_VALIDATION_FREQ_LO_HZ),
            freq_hi=float(audit.B3_CISS_VALIDATION_FREQ_HI_HZ),
        )
        payload["B3_Lmid_CISS_converged_mode_count"] = int(nconv)
        payload["B3_Lmid_CISS_accepted_mode_count"] = int(
            sum(1 for i in range(nconv) if payload.get(f"B3_DEV_mode_{i}_acceptance_pass"))
        )
        ciss_freqs = _accepted_frequencies_from_mode_payload(payload)
        payload["B3_Lmid_CISS_accepted_frequencies_hz"] = ciss_freqs
        payload["B3_Lmid_CISS_per_mode_diagnostics"] = _per_mode_diagnostics_from_payload(payload, nconv=nconv)
        payload.update(_modal_facet_proxy_export_note())
        payload["B3_Lmid_CISS_solve_elapsed_seconds"] = dev_bench._safe_float(
            float(payload.get("B3_DEV_timing_eps_solve_end_elapsed_seconds", 0))
            - float(payload.get("B3_DEV_timing_eps_solve_begin_elapsed_seconds", 0))
        )
        timer_ciss.finalize()
        payload["B3_Lmid_CISS_total_elapsed_seconds"] = payload.get("B3_DEV_timing_total_wall_elapsed_seconds")
        ciss_ok = bool(accepted or nconv > 0)
        verdict = "B3_Lmid_OVERNIGHT_CISS_REFERENCE_PASS" if ciss_ok else "B3_Lmid_OVERNIGHT_CISS_REFERENCE_FAILED"
        payload["CISS_reference_available"] = bool(ciss_ok)
        payload["reference_available"] = bool(ciss_ok)
    except Exception as exc:
        payload["B3_Lmid_failure_reason"] = f"{type(exc).__name__}:{exc}"
        ciss_ok = False
    finally:
        if eps is not None:
            try:
                eps.destroy()
            except Exception:
                pass
        payload["next_step_verdict"] = verdict
        _write_report(OUT_JSON_LMID_CISS, OUT_MD_LMID_CISS, payload, title="L_mid overnight CISS reference")
    return payload, ciss_ok, ciss_freqs


def _run_lmid_st_multi_target(
    *,
    built: Dict[str, Any],
    targets_hz: List[float],
) -> Tuple[Dict[str, Any], List[float], bool]:
    from slepc4py import SLEPc

    freq_lo = float(audit.B3_CISS_VALIDATION_FREQ_LO_HZ)
    freq_hi = float(audit.B3_CISS_VALIDATION_FREQ_HI_HZ)
    payload: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "B3_Lmid_overnight_ST_multi_target_only",
        "B3_Lmid_mesh_level": LMID_MESH_LEVEL,
        "B3_Lmid_ST_multi_target_targets_hz": list(targets_hz),
        "B3_Lmid_ST_multi_target_operator_built_once_pass": True,
        "B3_Lmid_solver_name": "KRYLOVSCHUR-ST-SINVERT-MUMPS-MULTI-TARGET",
        "no_new_eigensolve_executed": True,
        "new_eigensolve_executed": False,
        "production_promotion": "BLOCKED",
    }
    timer = dev_bench._B3DevTiming(payload)
    A_active = built["A_active"]
    M_active = built["M_active"]
    all_accepted: List[Dict[str, Any]] = []
    total_setup_s = 0.0
    total_solve_s = 0.0
    any_solve_ok = False
    union_freqs: List[float] = []
    verdict = "B3_Lmid_OVERNIGHT_ST_MULTI_TARGET_BLOCKED"

    try:
        for ti, target_hz in enumerate(targets_hz):
            prefix = f"B3_Lmid_ST_multi_target_target_{ti}_"
            payload[f"{prefix}target_frequency_hz"] = float(target_hz)
            payload[f"{prefix}target_lambda"] = dev_bench._safe_float(
                audit._b3_hz_to_lambda_sq(float(target_hz))
            )
            eps = None
            try:
                eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
                cfg: Dict[str, Any] = {}
                dev_bench._dev_configure_coarse_krylovschur_sinvert_eps(
                    eps,
                    A_active,
                    M_active,
                    payload=cfg,
                    target_lambda=float(audit._b3_hz_to_lambda_sq(float(target_hz))),
                    target_hz=float(target_hz),
                )
                t0 = time.perf_counter()
                eps.setUp()
                setup_s = time.perf_counter() - t0
                intro = dev_bench._dev_introspect_st_targeting_after_setup(eps)
                payload[f"{prefix}effective_target"] = intro.get("B3_DEV_ST_target_effective")
                payload[f"{prefix}effective_shift"] = intro.get("B3_DEV_ST_shift_effective")
                payload[f"{prefix}effective_which"] = intro.get("B3_DEV_ST_which_effective_normalized")
                payload[f"{prefix}setup_elapsed_seconds"] = dev_bench._safe_float(setup_s)

                t1 = time.perf_counter()
                eps.solve()
                solve_s = time.perf_counter() - t1
                payload[f"{prefix}solve_elapsed_seconds"] = dev_bench._safe_float(solve_s)
                payload[f"{prefix}solve_pass"] = True
                total_setup_s += setup_s
                total_solve_s += solve_s
                any_solve_ok = True

                nconv, accepted = dev_bench._dev_collect_accepted_st_modes(
                    eps,
                    A_active,
                    built,
                    target_hz=float(target_hz),
                    freq_lo=freq_lo,
                    freq_hi=freq_hi,
                )
                payload[f"{prefix}converged_mode_count"] = int(nconv)
                payload[f"{prefix}accepted_mode_count_in_interval"] = int(len(accepted))
                acc_f = [float(m["frequency_hz"]) for m in accepted]
                payload[f"{prefix}accepted_frequencies"] = acc_f
                payload[f"{prefix}accepted_eps_relative_errors"] = [
                    dev_bench._safe_float(m.get("eps_compute_error_relative")) for m in accepted
                ]
                all_accepted.extend(accepted)
            except Exception as exc:
                payload[f"{prefix}solve_pass"] = False
                payload[f"{prefix}failure_reason"] = f"{type(exc).__name__}:{exc}"
            finally:
                if eps is not None:
                    try:
                        eps.destroy()
                    except Exception:
                        pass

        payload["new_eigensolve_executed"] = bool(any_solve_ok)
        payload["no_new_eigensolve_executed"] = not any_solve_ok
        payload["B3_Lmid_ST_multi_target_total_setup_elapsed_seconds"] = dev_bench._safe_float(total_setup_s)
        payload["B3_Lmid_ST_multi_target_total_solve_elapsed_seconds"] = dev_bench._safe_float(total_solve_s)
        union_freqs = dev_bench._dev_deduplicate_frequencies_hz(
            [float(m["frequency_hz"]) for m in all_accepted],
            tol_hz=dev_bench.B3_DEV_ST_MULTI_DEDUP_TOL_HZ,
        )
        payload["B3_Lmid_ST_multi_target_unique_accepted_frequency_count"] = int(len(union_freqs))
        payload["B3_Lmid_ST_multi_target_unique_accepted_frequencies"] = union_freqs
        payload["B3_Lmid_ST_multi_target_deduplicated_mode_provenance"] = _st_deduplicated_provenance(
            all_accepted,
            union_freqs,
            tol_hz=dev_bench.B3_DEV_ST_MULTI_DEDUP_TOL_HZ,
        )
        payload["B3_Lmid_ST_per_target_accepted_mode_records"] = list(all_accepted)
        try:
            payload.update(_modal_facet_proxy_export_note())
        except Exception as exc:
            payload["B3_Lmid_modal_observables_optional_export_pass"] = False
            payload["B3_Lmid_modal_observables_optional_export_failure"] = f"{type(exc).__name__}:{exc}"
        timer.finalize()
        payload["B3_Lmid_ST_multi_target_total_elapsed_seconds"] = payload.get("B3_DEV_timing_total_wall_elapsed_seconds")
        verdict = (
            "B3_Lmid_OVERNIGHT_ST_MULTI_TARGET_PASS"
            if any_solve_ok and union_freqs
            else "B3_Lmid_OVERNIGHT_ST_MULTI_TARGET_COMPLETED"
        )
    except Exception as exc:
        payload["B3_Lmid_failure_reason"] = f"{type(exc).__name__}:{exc}"
        union_freqs = []
    finally:
        payload["next_step_verdict"] = verdict
        _write_report(OUT_JSON_LMID_ST, OUT_MD_LMID_ST, payload, title="L_mid overnight ST multi-target")
    return payload, union_freqs, any_solve_ok


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _lmid_st_ciss_comparison_block(
    *,
    st_freqs: List[float],
    st_payload: Dict[str, Any],
    ciss_freqs: List[float],
    ciss_ok: bool,
    ciss_payload: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """ST-vs-CISS summary fields and checkpoint payload (no solver execution)."""
    summary_fields: Dict[str, Any] = {
        "B3_Lmid_CISS_reference_available": bool(ciss_ok),
        "CISS_reference_available": bool(ciss_ok),
        "B3_Lmid_CISS_accepted_frequency_count": len(ciss_freqs),
        "B3_Lmid_CISS_accepted_frequencies_hz": ciss_freqs,
        "B3_Lmid_CISS_setup_elapsed_seconds": ciss_payload.get("B3_Lmid_CISS_setup_elapsed_seconds"),
        "B3_Lmid_CISS_solve_elapsed_seconds": ciss_payload.get("B3_Lmid_CISS_solve_elapsed_seconds"),
        "B3_Lmid_CISS_total_elapsed_seconds": ciss_payload.get("B3_Lmid_CISS_total_elapsed_seconds"),
        "ST_results_available": bool(st_freqs),
        "B3_Lmid_ST_multi_target_unique_accepted_frequency_count": len(st_freqs),
        "B3_Lmid_ST_multi_target_unique_accepted_frequencies": st_freqs,
        "B3_Lmid_ST_multi_target_total_setup_elapsed_seconds": st_payload.get(
            "B3_Lmid_ST_multi_target_total_setup_elapsed_seconds"
        ),
        "B3_Lmid_ST_multi_target_total_solve_elapsed_seconds": st_payload.get(
            "B3_Lmid_ST_multi_target_total_solve_elapsed_seconds"
        ),
        "B3_Lmid_ST_multi_target_total_elapsed_seconds": st_payload.get(
            "B3_Lmid_ST_multi_target_total_elapsed_seconds"
        ),
    }
    st_ciss_ckpt: Dict[str, Any] = {
        "mode": "B3_Lmid_overnight_ST_vs_CISS_comparison_checkpoint",
        "B3_Lmid_ST_multi_target_unique_accepted_frequencies": st_freqs,
        "B3_Lmid_ST_multi_target_deduplicated_mode_provenance": st_payload.get(
            "B3_Lmid_ST_multi_target_deduplicated_mode_provenance"
        ),
    }
    if ciss_ok and ciss_freqs:
        matches, missing, extra = dev_bench._dev_compare_frequency_sets(
            st_freqs,
            ciss_freqs,
            match_tol_hz=dev_bench.B3_DEV_ST_MULTI_CISS_MATCH_TOL_HZ,
        )
        summary_fields["B3_Lmid_ST_multi_target_CISS_reference_frequency_count"] = len(ciss_freqs)
        summary_fields["B3_Lmid_ST_multi_target_matches_CISS_count"] = int(matches)
        summary_fields["B3_Lmid_ST_multi_target_missing_CISS_frequencies"] = missing
        summary_fields["B3_Lmid_ST_multi_target_extra_frequencies_in_interval"] = extra
        summary_fields["B3_Lmid_ST_multi_target_full_interval_coverage_pass"] = bool(
            len(missing) == 0 and matches == len(ciss_freqs)
        )
        summary_fields["coverage_comparison_unavailable_due_to_CISS_failure"] = False
        st_ciss_ckpt.update(
            {
                "B3_Lmid_ST_multi_target_matches_CISS_count": int(matches),
                "B3_Lmid_ST_multi_target_missing_CISS_frequencies": missing,
                "B3_Lmid_ST_multi_target_extra_frequencies_in_interval": extra,
                "B3_Lmid_ST_multi_target_full_interval_coverage_pass": summary_fields[
                    "B3_Lmid_ST_multi_target_full_interval_coverage_pass"
                ],
            }
        )
        ciss_total = summary_fields.get("B3_Lmid_CISS_total_elapsed_seconds")
        st_total = summary_fields.get("B3_Lmid_ST_multi_target_total_elapsed_seconds")
        if (
            ciss_total is not None
            and st_total is not None
            and math.isfinite(float(ciss_total))
            and math.isfinite(float(st_total))
            and float(st_total) > 0.0
        ):
            summary_fields["B3_Lmid_ST_multi_target_speedup_vs_CISS"] = dev_bench._safe_float(
                float(ciss_total) / float(st_total)
            )
            st_ciss_ckpt["B3_Lmid_ST_multi_target_speedup_vs_CISS"] = summary_fields[
                "B3_Lmid_ST_multi_target_speedup_vs_CISS"
            ]
    else:
        summary_fields["B3_Lmid_ST_multi_target_CISS_reference_frequency_count"] = 0
        summary_fields["B3_Lmid_ST_multi_target_matches_CISS_count"] = None
        summary_fields["B3_Lmid_ST_multi_target_missing_CISS_frequencies"] = None
        summary_fields["B3_Lmid_ST_multi_target_extra_frequencies_in_interval"] = None
        summary_fields["B3_Lmid_ST_multi_target_full_interval_coverage_pass"] = None
        summary_fields["coverage_comparison_unavailable_due_to_CISS_failure"] = True
        st_ciss_ckpt["coverage_comparison_unavailable_due_to_CISS_failure"] = True
    st_ciss_ckpt["next_step_verdict"] = (
        "B3_Lmid_OVERNIGHT_ST_vs_CISS_COMPARISON_RECORDED"
        if st_ciss_ckpt.get("B3_Lmid_ST_multi_target_full_interval_coverage_pass")
        else "B3_Lmid_OVERNIGHT_ST_vs_CISS_COMPARISON_PENDING_OR_INCOMPLETE"
    )
    return summary_fields, st_ciss_ckpt


def _lmid_finalize_st_ciss_summary_verdict(summary: Dict[str, Any], *, st_ok: bool, ciss_ok: bool) -> Tuple[str, int]:
    if summary.get("B3_Lmid_operator_contract_pass") and st_ok:
        if ciss_ok and summary.get("B3_Lmid_ST_multi_target_full_interval_coverage_pass"):
            return "B3_Lmid_OVERNIGHT_CISS_ST_VALIDATION_PASS", 0
        if ciss_ok and not summary.get("B3_Lmid_ST_multi_target_full_interval_coverage_pass"):
            return "B3_Lmid_OVERNIGHT_ST_COVERAGE_INCOMPLETE", 2
        if not ciss_ok and st_ok:
            return "B3_Lmid_OVERNIGHT_ST_PASS_CISS_REFERENCE_FAILED", 2
        return "B3_Lmid_OVERNIGHT_VALIDATION_PARTIAL", 2
    if st_ok and ciss_ok and summary.get("B3_Lmid_ST_multi_target_full_interval_coverage_pass"):
        return "B3_Lmid_OVERNIGHT_CISS_ST_VALIDATION_PASS", 0
    if st_ok and ciss_ok:
        return "B3_Lmid_OVERNIGHT_ST_COVERAGE_INCOMPLETE", 2
    if st_ok and not ciss_ok:
        return "B3_Lmid_OVERNIGHT_ST_PASS_CISS_REFERENCE_FAILED", 2
    return "B3_Lmid_OVERNIGHT_VALIDATION_BLOCKED", 2


def run_lmid_ciss_reference_only(pre: Dict[str, Any]) -> int:
    """Build L_mid corrected operator once and run CISS reference on [220,265] Hz only."""
    if not pre.get("preassembly_contract_pass") or MPI.COMM_WORLD.size != 1:
        return 2
    if not _lmid_mesh_path().is_file():
        print(f"[B3_Lmid] mesh_missing={_lmid_mesh_path()}", flush=True)
        return 2

    mats: List[Any] = []
    seen: set[int] = set()
    rc = 2
    try:
        op_payload: Dict[str, Any] = {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": "B3_Lmid_overnight_CISS_reference_only_operator_contract",
        }
        timer = dev_bench._B3DevTiming(op_payload)
        timer.mark("operator_build_begin")
        built = audit._b3_build_corrected_structural_active_operators(
            mats_to_destroy=mats,
            mat_destroy_seen=seen,
            mesh_level=LMID_MESH_LEVEL,
            struct_active_count_policy="L_mid_exact",
        )
        timer.mark("operator_build_end")
        op_payload["B3_Lmid_operator_build_elapsed_seconds"] = op_payload.get(
            "B3_DEV_timing_operator_build_end_elapsed_seconds"
        )
        if not _lmid_operator_contract_pass(op_payload, built=built):
            op_payload["B3_Lmid_failure_reason"] = "operator_contract_failed"
            op_payload["next_step_verdict"] = "B3_Lmid_OVERNIGHT_OPERATOR_CONTRACT_BLOCKED"
            _write_checkpoint(OUT_JSON_LMID_CONTRACT_CKPT, op_payload)
            return 2
        _write_checkpoint(
            OUT_JSON_LMID_CONTRACT_CKPT,
            {
                "mode": "B3_Lmid_overnight_operator_contract_checkpoint",
                "next_step_verdict": "B3_Lmid_OVERNIGHT_OPERATOR_CONTRACT_PASS",
                **op_payload,
            },
        )
        _, ciss_ok, _ = _run_lmid_ciss_reference(built=built)
        rc = 0 if ciss_ok else 2
    finally:
        audit._destroy_mats_deduped(mats)
    return rc


def run_lmid_st_ciss_comparison_only() -> int:
    """Load existing L_mid ST + CISS JSON artifacts and emit ST-vs-CISS summary (no solver)."""
    st_payload = _load_json_if_exists(OUT_JSON_LMID_ST)
    ciss_payload = _load_json_if_exists(OUT_JSON_LMID_CISS)
    if st_payload is None:
        print(f"[B3_Lmid] missing_ST_json={OUT_JSON_LMID_ST}", flush=True)
        return 2
    if ciss_payload is None:
        print(f"[B3_Lmid] missing_CISS_json={OUT_JSON_LMID_CISS}", flush=True)
        return 2

    st_freqs = list(st_payload.get("B3_Lmid_ST_multi_target_unique_accepted_frequencies") or [])
    ciss_freqs = list(ciss_payload.get("B3_Lmid_CISS_accepted_frequencies_hz") or [])
    if not ciss_freqs:
        ciss_freqs = _accepted_frequencies_from_mode_payload(ciss_payload)
    ciss_ok = bool(
        ciss_payload.get("CISS_reference_available")
        or ciss_payload.get("reference_available")
        or str(ciss_payload.get("next_step_verdict") or "").endswith("_PASS")
    )

    summary: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "B3_Lmid_overnight_ST_vs_CISS_comparison_only_summary",
        "B3_Lmid_mesh_level": LMID_MESH_LEVEL,
        "B3_Lmid_mesh_path": str(_lmid_mesh_path().resolve()),
        "B3_Lmid_ST_multi_target_targets_hz": list(LMID_ST_TARGETS_HZ),
        "production_promotion": "BLOCKED",
        "no_automatic_production_promotion": True,
        "comparison_only_no_solver_execution": True,
    }
    existing = _load_json_if_exists(OUT_JSON_LMID_SUMMARY) or _load_json_if_exists(OUT_JSON_LMID_CONTRACT_CKPT)
    if existing:
        for key in (
            "B3_Lmid_active_dimension",
            "B3_Lmid_active_dimension_expected",
            "B3_Lmid_active_dimension_contract_pass",
            "B3_Lmid_A_shape",
            "B3_Lmid_M_shape",
            "B3_Lmid_operator_contract_pass",
            "B3_Lmid_operator_build_elapsed_seconds",
        ):
            if key in existing:
                summary[key] = existing[key]

    comparison_fields, st_ciss_ckpt = _lmid_st_ciss_comparison_block(
        st_freqs=st_freqs,
        st_payload=st_payload,
        ciss_freqs=ciss_freqs,
        ciss_ok=ciss_ok,
        ciss_payload=ciss_payload,
    )
    summary.update(comparison_fields)
    summary["B3_Lmid_CISS_per_mode_diagnostics"] = ciss_payload.get("B3_Lmid_CISS_per_mode_diagnostics")
    summary.update(_modal_facet_proxy_export_note())

    st_ok = bool(st_freqs) or str(st_payload.get("next_step_verdict") or "").endswith("_PASS")
    verdict, rc = _lmid_finalize_st_ciss_summary_verdict(summary, st_ok=st_ok, ciss_ok=ciss_ok)
    summary["next_step_verdict"] = verdict
    _write_checkpoint(OUT_JSON_LMID_ST_CISS_CKPT, st_ciss_ckpt)
    _write_report(
        OUT_JSON_LMID_SUMMARY,
        OUT_MD_LMID_SUMMARY,
        summary,
        title="L_mid ST vs CISS comparison summary",
    )
    return rc


def run_lmid_overnight_validation(pre: Dict[str, Any]) -> int:
    if not pre.get("preassembly_contract_pass") or MPI.COMM_WORLD.size != 1:
        return 2
    if not _lmid_mesh_path().is_file():
        print(f"[B3_Lmid] mesh_missing={_lmid_mesh_path()}", flush=True)
        return 2

    summary: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "B3_Lmid_overnight_CISS_ST_multi_target_validation_summary",
        "B3_Lmid_mesh_level": LMID_MESH_LEVEL,
        "B3_Lmid_mesh_path": str(_lmid_mesh_path().resolve()),
        "B3_Lmid_ST_multi_target_targets_hz": list(LMID_ST_TARGETS_HZ),
        "production_promotion": "BLOCKED",
        "no_automatic_production_promotion": True,
    }
    timer = dev_bench._B3DevTiming(summary)
    mats: List[Any] = []
    seen: set[int] = set()
    verdict = "B3_Lmid_OVERNIGHT_VALIDATION_BLOCKED"
    rc = 2

    try:
        timer.mark("operator_build_begin")
        built = audit._b3_build_corrected_structural_active_operators(
            mats_to_destroy=mats,
            mat_destroy_seen=seen,
            mesh_level=LMID_MESH_LEVEL,
            struct_active_count_policy="L_mid_exact",
        )
        timer.mark("operator_build_end")
        summary["B3_Lmid_operator_build_elapsed_seconds"] = summary.get(
            "B3_DEV_timing_operator_build_end_elapsed_seconds"
        )
        op_payload: Dict[str, Any] = {}
        if not _lmid_operator_contract_pass(op_payload, built=built):
            summary.update(op_payload)
            summary["B3_Lmid_failure_reason"] = "operator_contract_failed"
            summary["next_step_verdict"] = "B3_Lmid_OVERNIGHT_OPERATOR_CONTRACT_BLOCKED"
            _write_checkpoint(OUT_JSON_LMID_CONTRACT_CKPT, dict(summary, **op_payload))
            _write_report(OUT_JSON_LMID_SUMMARY, OUT_MD_LMID_SUMMARY, summary, title="L_mid overnight summary")
            return 2
        summary.update(op_payload)
        _write_checkpoint(
            OUT_JSON_LMID_CONTRACT_CKPT,
            {
                "mode": "B3_Lmid_overnight_operator_contract_checkpoint",
                "next_step_verdict": "B3_Lmid_OVERNIGHT_OPERATOR_CONTRACT_PASS",
                **op_payload,
            },
        )

        ciss_payload, ciss_ok, ciss_freqs = _run_lmid_ciss_reference(built=built)

        st_payload, st_freqs, st_ok = _run_lmid_st_multi_target(built=built, targets_hz=LMID_ST_TARGETS_HZ)
        comparison_fields, st_ciss_ckpt = _lmid_st_ciss_comparison_block(
            st_freqs=st_freqs,
            st_payload=st_payload,
            ciss_freqs=ciss_freqs,
            ciss_ok=ciss_ok,
            ciss_payload=ciss_payload,
        )
        summary.update(comparison_fields)
        _write_checkpoint(OUT_JSON_LMID_ST_CISS_CKPT, st_ciss_ckpt)

        dense_ciss, dense_st = _load_dense_reference_payloads()
        dense_ciss_freqs = _accepted_frequencies_from_mode_payload(dense_ciss)
        dense_st_freqs = list(dense_st.get("B3_DEV_ST_multi_target_unique_accepted_frequencies") or [])
        ref_freqs = ciss_freqs if ciss_freqs else st_freqs
        cross_mesh = _cross_mesh_convergence_report(
            lmid_freqs=ref_freqs,
            dense_ciss_freqs=dense_ciss_freqs,
            dense_st_freqs=dense_st_freqs,
        )
        summary.update(cross_mesh)
        _write_checkpoint(
            OUT_JSON_LMID_CROSS_MESH_CKPT,
            {"mode": "B3_Lmid_overnight_cross_mesh_comparison_checkpoint", **cross_mesh},
        )
        summary["B3_Lmid_CISS_per_mode_diagnostics"] = ciss_payload.get("B3_Lmid_CISS_per_mode_diagnostics")
        summary.update(_modal_facet_proxy_export_note())

        timer.finalize()
        summary["B3_Lmid_overnight_total_elapsed_seconds"] = summary.get("B3_DEV_timing_total_wall_elapsed_seconds")

        if summary.get("B3_Lmid_operator_contract_pass") and st_ok:
            if ciss_ok and summary.get("B3_Lmid_ST_multi_target_full_interval_coverage_pass"):
                verdict = "B3_Lmid_OVERNIGHT_CISS_ST_VALIDATION_PASS"
                rc = 0
            elif ciss_ok and not summary.get("B3_Lmid_ST_multi_target_full_interval_coverage_pass"):
                verdict = "B3_Lmid_OVERNIGHT_ST_COVERAGE_INCOMPLETE"
                rc = 2
            elif not ciss_ok and st_ok:
                verdict = "B3_Lmid_OVERNIGHT_ST_PASS_CISS_REFERENCE_FAILED"
                rc = 2
            else:
                verdict = "B3_Lmid_OVERNIGHT_VALIDATION_PARTIAL"
                rc = 2
        else:
            verdict = "B3_Lmid_OVERNIGHT_VALIDATION_BLOCKED"
            rc = 2
    except Exception as exc:
        summary["B3_Lmid_failure_reason"] = f"{type(exc).__name__}:{exc}"
        rc = 2
    finally:
        if "B3_DEV_timing_total_wall_elapsed_seconds" not in summary:
            timer.finalize()
        summary["next_step_verdict"] = verdict
        _write_report(
            OUT_JSON_LMID_SUMMARY,
            OUT_MD_LMID_SUMMARY,
            summary,
            title="L_mid overnight CISS+ST validation summary",
        )
        audit._destroy_mats_deduped(mats)
    return rc
