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
B3_DEV_COARSE_ST_TARGETING_PREFLIGHT_ARG = "--B3-dev-coarse-krylovschur-sinvert-targeting-preflight-only"
B3_DEV_REFINED_ST_MULTI_TARGET_ARG = (
    "--B3-dev-refined-krylovschur-sinvert-multi-target-coverage-benchmark-only"
)
B3_DEV_ST_MULTI_TARGETS_HZ_ARG = "--B3-dev-st-multi-targets-hz"
B3_DEV_DENSE_COVERAGE_COMPARE_ARG = "--B3-dev-dense-st-ciss-coverage-compare-only"
B3_DEV_COMPARE_ARG = "--B3-dev-solver-benchmark-compare-only"

B3_DEV_DEFAULT_MESH_VARIANT = "L_dev_coarse"
B3_DEV_ALLOWED_MESH_VARIANTS = frozenset({"L_dev_coarse", "L_dev_refined", "L_dev_dense"})
B3_DEV_COARSE_NEV = 12
B3_DEV_COARSE_NCV = 24

_DEV_JSON_STEM_CONTRACT = "v2_B3_DEV_coarse_corrected_operator_contract_only"
_DEV_JSON_STEM_CISS = "v2_B3_DEV_coarse_ciss_direct_stable_benchmark_only"
_DEV_JSON_STEM_ST = "v2_B3_DEV_coarse_krylovschur_sinvert_benchmark_only"
_DEV_JSON_STEM_ST_TARGETING = "v2_B3_DEV_coarse_krylovschur_sinvert_targeting_preflight_only"
_DEV_JSON_STEM_ST_MULTI_TARGET = "v2_B3_DEV_krylovschur_sinvert_multi_target_coverage_benchmark_only"
_DEV_JSON_STEM_SUMMARY = "v2_B3_DEV_coarse_solver_benchmark_summary"

B3_DEV_ST_MULTI_TARGET_FREQS_HZ = [224.0, 244.39, 262.5]
B3_DEV_ST_MULTI_DEDUP_TOL_HZ = 1.0e-3
B3_DEV_ST_MULTI_CISS_MATCH_TOL_HZ = 0.05


def _dev_out_json(stem: str, mesh_variant: str) -> Path:
    """L_dev_coarse keeps legacy filenames; other variants get a mesh suffix."""
    if mesh_variant == "L_dev_coarse":
        return audit.CONV_DIAG / f"{stem}.json"
    return audit.CONV_DIAG / f"{stem}_{mesh_variant}.json"


def _dev_out_md(stem: str, mesh_variant: str) -> Path:
    return _dev_out_json(stem, mesh_variant).with_suffix(".md")


def _dev_st_multi_targets_default() -> List[float]:
    return list(B3_DEV_ST_MULTI_TARGET_FREQS_HZ)


def _dev_st_multi_targets_equal(a: List[float], b: List[float], *, rtol: float = 1.0e-9) -> bool:
    if len(a) != len(b):
        return False
    return all(abs(float(x) - float(y)) <= rtol * max(1.0, abs(float(y))) for x, y in zip(a, b))


def _dev_st_multi_target_stem_suffix(targets_hz: List[float]) -> str:
    """Distinct JSON stem suffix when targets differ from default three-target list."""
    if _dev_st_multi_targets_equal(targets_hz, _dev_st_multi_targets_default()):
        return ""
    parts: List[str] = []
    for f in targets_hz:
        s = f"{float(f):g}".replace(".", "p")
        parts.append(s)
    return "_hz_" + "_".join(parts)


def _dev_out_json_st_multi_target(mesh_variant: str, targets_hz: List[float]) -> Path:
    stem = _DEV_JSON_STEM_ST_MULTI_TARGET + _dev_st_multi_target_stem_suffix(targets_hz)
    return _dev_out_json(stem, mesh_variant)


def _parse_hz_list(text: str) -> List[float]:
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


def _parse_dev_st_multi_targets_hz(argv: List[str]) -> List[float]:
    for i, arg in enumerate(argv):
        if arg == B3_DEV_ST_MULTI_TARGETS_HZ_ARG and i + 1 < len(argv):
            return _parse_hz_list(argv[i + 1])
        if arg.startswith(f"{B3_DEV_ST_MULTI_TARGETS_HZ_ARG}="):
            return _parse_hz_list(arg.split("=", 1)[1])
    return _dev_st_multi_targets_default()


def _dev_mesh_active_dimension_targets(mesh_variant: str) -> Tuple[Optional[int], Optional[int]]:
    from v2_mesh_convergence_common import load_manifest

    level_def = (load_manifest().get("mesh_levels") or {}).get(mesh_variant) or {}
    lo = level_def.get("target_active_dimension_min")
    hi = level_def.get("target_active_dimension_max")
    return (
        int(lo) if lo is not None else None,
        int(hi) if hi is not None else None,
    )


def _dev_active_dimension_in_target_range(mesh_variant: str, active_dim: int) -> bool:
    lo, hi = _dev_mesh_active_dimension_targets(mesh_variant)
    if lo is None or hi is None:
        return True
    return int(lo) <= int(active_dim) <= int(hi)


def _dev_record_active_dimension_target_range(payload: Dict[str, Any], mesh_variant: str) -> Optional[str]:
    active_dim = int(payload.get("B3_DEV_active_dimension") or 0)
    dim_lo, dim_hi = _dev_mesh_active_dimension_targets(mesh_variant)
    payload["B3_DEV_active_dimension_target_min"] = dim_lo
    payload["B3_DEV_active_dimension_target_max"] = dim_hi
    payload["B3_DEV_active_dimension_in_target_range_pass"] = _dev_active_dimension_in_target_range(
        mesh_variant, active_dim
    )
    if not payload["B3_DEV_active_dimension_in_target_range_pass"]:
        return f"active_dimension={active_dim}_outside_target_{dim_lo}_{dim_hi}"
    return None


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


def _dev_tag_count(tag_map: Any, tag: int) -> int:
    if not isinstance(tag_map, dict):
        return 0
    return int(tag_map.get(str(tag), tag_map.get(tag, 0)) or 0)


def _dev_mesh_bootstrap_payload(mesh_variant: str) -> Dict[str, Any]:
    from v2_mesh_convergence_common import load_manifest, mesh_audit_path

    mesh_file = audit.mesh_path(mesh_variant, audit.CASE_ID)
    manifest = load_manifest()
    level_def = (manifest.get("mesh_levels") or {}).get(mesh_variant) or {}
    lc_scale = level_def.get("lc_scale")
    node_count = None
    element_count = None
    volume_tags_ok = None
    facet_tags_ok = None
    soundhole_facets_tag2 = None
    audit_json = mesh_audit_path(mesh_variant, audit.CASE_ID)
    if audit_json.is_file():
        try:
            ad = json.loads(audit_json.read_text(encoding="utf-8"))
            node_count = ad.get("n_nodes")
            element_count = ad.get("n_tetrahedra")
            if lc_scale is None:
                lc_scale = ad.get("lc_scale")
            vol = ad.get("volume_tag_counts") or {}
            tri = ad.get("triangle_tag_counts") or {}
            volume_tags_ok = all(_dev_tag_count(vol, t) > 0 for t in (1, 2, 3, 10))
            facet_tags_ok = all(_dev_tag_count(tri, t) > 0 for t in (1, 2, 3, 4, 5))
            soundhole_facets_tag2 = _dev_tag_count(tri, 2)
        except Exception:
            pass
    build_summary = audit.CONV_DIAG.parent / "mesh" / mesh_variant / "baseline_coupled_v2_mesh_build_summary.json"
    if build_summary.is_file():
        try:
            bs = json.loads(build_summary.read_text(encoding="utf-8"))
            if volume_tags_ok is None:
                volume_tags_ok = bs.get("volume_tags_ok")
            if facet_tags_ok is None:
                facet_tags_ok = bs.get("facet_tags_ok")
            if soundhole_facets_tag2 is None:
                soundhole_facets_tag2 = bs.get("soundhole_facets_tag2")
            if lc_scale is None:
                lc_scale = bs.get("B3_DEV_mesh_lc_scale") or bs.get("lc_scale")
            if node_count is None:
                node_count = bs.get("B3_DEV_mesh_node_count") or bs.get("n_nodes")
            if element_count is None:
                element_count = bs.get("B3_DEV_mesh_element_count") or bs.get("n_tetrahedra")
        except Exception:
            pass
    return {
        "B3_DEV_mesh_variant": mesh_variant,
        "B3_DEV_mesh_is_solver_smoke_test_only": True,
        "B3_DEV_mesh_not_authorized_for_final_physics_validation": True,
        "B3_DEV_mesh_path": str(mesh_file.resolve()),
        "B3_DEV_mesh_lc_scale": _safe_float(lc_scale) if lc_scale is not None else None,
        "B3_DEV_mesh_node_count": node_count,
        "B3_DEV_mesh_element_count": element_count,
        "volume_tags_ok": volume_tags_ok,
        "facet_tags_ok": facet_tags_ok,
        "soundhole_facets_tag2": soundhole_facets_tag2,
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
    payload: Dict[str, Any],
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


def _dev_collect_accepted_st_modes(
    eps: Any,
    A_active: Any,
    built: Dict[str, Any],
    *,
    target_hz: float,
    freq_lo: float,
    freq_hi: float,
) -> Tuple[int, List[Dict[str, Any]]]:
    """Return converged count and accepted modes in [freq_lo, freq_hi] (no global payload keys)."""
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
            if mode_pass:
                accepted.append(
                    {
                        "mode_index": i,
                        "frequency_hz": float(f_hz),
                        "lambda_real": lam_re,
                        "lambda_imag": lam_im,
                        "eps_compute_error_relative": eps_err,
                        "st_shift_target_hz": float(target_hz),
                    }
                )
        finally:
            vr.destroy()
            vi.destroy()
    return nconv, accepted


def _dev_deduplicate_frequencies_hz(freqs: List[float], *, tol_hz: float) -> List[float]:
    if not freqs:
        return []
    sorted_f = sorted(float(f) for f in freqs)
    out = [sorted_f[0]]
    for f in sorted_f[1:]:
        if abs(f - out[-1]) > tol_hz:
            out.append(f)
    return out


def _dev_load_ciss_reference_accepted_frequencies(
    mesh_variant: str,
    *,
    freq_lo: float,
    freq_hi: float,
) -> Tuple[bool, List[float], Optional[Dict[str, Any]]]:
    path = _dev_out_json(_DEV_JSON_STEM_CISS, mesh_variant)
    if not path.is_file():
        return False, [], None
    data = json.loads(path.read_text(encoding="utf-8"))
    freqs: List[float] = []
    nconv = int(data.get("B3_DEV_converged_mode_count") or 0)
    for i in range(nconv):
        if not data.get(f"B3_DEV_mode_{i}_acceptance_pass"):
            continue
        f = data.get(f"B3_DEV_mode_{i}_frequency_hz_if_real_positive")
        if f is None or not math.isfinite(float(f)):
            continue
        ff = float(f)
        if freq_lo <= ff <= freq_hi:
            freqs.append(ff)
    return True, sorted(freqs), data


def _dev_compare_frequency_sets(
    st_freqs: List[float],
    ciss_freqs: List[float],
    *,
    match_tol_hz: float,
) -> Tuple[int, List[float], List[float]]:
    """One-to-one greedy match: each CISS freq must have a distinct ST freq within tolerance."""
    st_pool = list(st_freqs)
    missing: List[float] = []
    matches = 0
    for f_ref in ciss_freqs:
        best_j = None
        best_d = None
        for j, f_st in enumerate(st_pool):
            d = abs(f_st - f_ref)
            if d <= match_tol_hz and (best_d is None or d < best_d):
                best_d = d
                best_j = j
        if best_j is None:
            missing.append(float(f_ref))
        else:
            matches += 1
            st_pool.pop(best_j)
    extra = list(st_pool)
    return matches, missing, extra


def _run_dev_st_multi_target_coverage_benchmark(
    pre: Dict[str, Any],
    mesh_variant: str,
    *,
    targets_hz: Optional[List[float]] = None,
) -> int:
    from slepc4py import SLEPc

    freq_lo = float(audit.B3_CISS_VALIDATION_FREQ_LO_HZ)
    freq_hi = float(audit.B3_CISS_VALIDATION_FREQ_HI_HZ)
    targets_hz = list(targets_hz if targets_hz is not None else _dev_st_multi_targets_default())
    out_json_path = _dev_out_json_st_multi_target(mesh_variant, targets_hz)
    payload: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "B3_dev_krylovschur_sinvert_multi_target_coverage_benchmark_only",
        "B3_DEV_solver_name": "KRYLOVSCHUR-ST-SINVERT-MUMPS-MULTI-TARGET",
        **_dev_mesh_bootstrap_payload(mesh_variant),
        "B3_DEV_validation_frequency_interval_hz": [freq_lo, freq_hi],
        "B3_DEV_ST_multi_target_targets_hz": targets_hz,
        "B3_DEV_ST_multi_target_output_json_path": str(out_json_path),
        "B3_DEV_ST_multi_target_operator_built_once_pass": False,
        "B3_DEV_ST_multi_target_solve_count": len(targets_hz),
        "B3_DEV_ST_multi_target_frequency_dedup_tol_hz": B3_DEV_ST_MULTI_DEDUP_TOL_HZ,
        "B3_DEV_ST_multi_target_CISS_match_tol_hz": B3_DEV_ST_MULTI_CISS_MATCH_TOL_HZ,
        "B3_DEV_execution_fallback_used": False,
        "B3_DEV_execution_automatic_retry_used": False,
        "no_new_eigensolve_executed": True,
        "new_eigensolve_executed": False,
        "production_promotion": "BLOCKED",
    }
    timer = _B3DevTiming(payload)
    mats: List[Any] = []
    seen: set[int] = set()
    verdict = "B3_DEV_ST_MULTI_TARGET_COVERAGE_BLOCKED"
    rc = 2
    try:
        if not pre.get("preassembly_contract_pass") or MPI.COMM_WORLD.size != 1:
            return 2

        ciss_path = _dev_out_json(_DEV_JSON_STEM_CISS, mesh_variant)
        ciss_loaded, ciss_ref_freqs, ciss_data = _dev_load_ciss_reference_accepted_frequencies(
            mesh_variant, freq_lo=freq_lo, freq_hi=freq_hi
        )
        payload["B3_DEV_ST_multi_target_CISS_reference_json_path"] = str(ciss_path)
        payload["B3_DEV_ST_multi_target_CISS_reference_loaded"] = bool(ciss_loaded and ciss_ref_freqs)
        payload["B3_DEV_ST_multi_target_CISS_coverage_comparison_available"] = bool(
            ciss_loaded and ciss_ref_freqs
        )
        if ciss_loaded and ciss_ref_freqs:
            payload["B3_DEV_ST_multi_target_CISS_reference_frequency_count"] = int(len(ciss_ref_freqs))
            payload["B3_DEV_ST_multi_target_CISS_reference_frequencies"] = ciss_ref_freqs
            payload["B3_DEV_ST_multi_target_CISS_reference_total_elapsed_seconds"] = _safe_float(
                (ciss_data or {}).get("B3_DEV_CISS_total_elapsed_seconds")
                or (ciss_data or {}).get("B3_DEV_timing_total_wall_elapsed_seconds")
            )
        else:
            payload["B3_DEV_ST_multi_target_CISS_reference_frequency_count"] = 0
            payload["B3_DEV_ST_multi_target_CISS_reference_frequencies"] = []
            payload["B3_DEV_ST_multi_target_CISS_reference_total_elapsed_seconds"] = None

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
        dim_fail = _dev_record_active_dimension_target_range(payload, mesh_variant)
        if dim_fail:
            payload["B3_DEV_failure_reason"] = dim_fail
            return 2
        payload["B3_DEV_ST_multi_target_operator_built_once_pass"] = True

        A_active = built["A_active"]
        M_active = built["M_active"]
        all_accepted_modes: List[Dict[str, Any]] = []
        total_setup_s = 0.0
        total_solve_s = 0.0

        for ti, target_hz in enumerate(targets_hz):
            target_lambda = float(audit._b3_hz_to_lambda_sq(target_hz))
            prefix = f"B3_DEV_ST_multi_target_target_{ti}_"
            payload[f"{prefix}target_frequency_hz"] = float(target_hz)
            payload[f"{prefix}target_lambda"] = _safe_float(target_lambda)

            eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
            try:
                cfg_payload: Dict[str, Any] = {}
                _dev_configure_coarse_krylovschur_sinvert_eps(
                    eps,
                    A_active,
                    M_active,
                    payload=cfg_payload,
                    target_lambda=target_lambda,
                    target_hz=float(target_hz),
                )
                t_setup0 = time.perf_counter()
                eps.setUp()
                setup_s = time.perf_counter() - t_setup0
                intro = _dev_introspect_st_targeting_after_setup(eps)
                payload[f"{prefix}effective_target"] = intro.get("B3_DEV_ST_target_effective")
                payload[f"{prefix}effective_shift"] = intro.get("B3_DEV_ST_shift_effective")
                payload[f"{prefix}effective_which"] = intro.get("B3_DEV_ST_which_effective_normalized")
                payload[f"{prefix}ST_type_effective"] = intro.get("B3_DEV_ST_ST_type_effective")
                payload[f"{prefix}setup_elapsed_seconds"] = _safe_float(setup_s)

                t_solve0 = time.perf_counter()
                eps.solve()
                solve_s = time.perf_counter() - t_solve0
                payload[f"{prefix}solve_elapsed_seconds"] = _safe_float(solve_s)
                total_setup_s += setup_s
                total_solve_s += solve_s

                nconv, accepted = _dev_collect_accepted_st_modes(
                    eps,
                    A_active,
                    built,
                    target_hz=float(target_hz),
                    freq_lo=freq_lo,
                    freq_hi=freq_hi,
                )
                payload[f"{prefix}converged_mode_count"] = int(nconv)
                payload[f"{prefix}accepted_mode_count_in_interval"] = int(len(accepted))
                acc_freqs = [float(m["frequency_hz"]) for m in accepted]
                payload[f"{prefix}accepted_frequencies"] = acc_freqs
                payload[f"{prefix}accepted_eps_relative_errors"] = [
                    _safe_float(m.get("eps_compute_error_relative")) for m in accepted
                ]
                all_accepted_modes.extend(accepted)
            finally:
                eps.destroy()

        payload["new_eigensolve_executed"] = True
        payload["no_new_eigensolve_executed"] = False
        payload["B3_DEV_ST_multi_target_total_setup_elapsed_seconds"] = _safe_float(total_setup_s)
        payload["B3_DEV_ST_multi_target_total_solve_elapsed_seconds"] = _safe_float(total_solve_s)

        union_freqs_raw = [float(m["frequency_hz"]) for m in all_accepted_modes]
        union_freqs = _dev_deduplicate_frequencies_hz(
            union_freqs_raw, tol_hz=B3_DEV_ST_MULTI_DEDUP_TOL_HZ
        )
        payload["B3_DEV_ST_multi_target_unique_accepted_frequency_count"] = int(len(union_freqs))
        payload["B3_DEV_ST_multi_target_unique_accepted_frequencies"] = union_freqs

        if payload["B3_DEV_ST_multi_target_CISS_coverage_comparison_available"]:
            matches, missing, extra = _dev_compare_frequency_sets(
                union_freqs,
                ciss_ref_freqs,
                match_tol_hz=B3_DEV_ST_MULTI_CISS_MATCH_TOL_HZ,
            )
            payload["B3_DEV_ST_multi_target_matches_CISS_count"] = int(matches)
            payload["B3_DEV_ST_multi_target_missing_CISS_frequencies"] = missing
            payload["B3_DEV_ST_multi_target_extra_frequencies_in_interval"] = extra
            payload["B3_DEV_ST_multi_target_full_interval_coverage_pass"] = bool(
                len(missing) == 0 and matches == len(ciss_ref_freqs)
            )
        else:
            payload["B3_DEV_ST_multi_target_matches_CISS_count"] = None
            payload["B3_DEV_ST_multi_target_missing_CISS_frequencies"] = None
            payload["B3_DEV_ST_multi_target_extra_frequencies_in_interval"] = None
            payload["B3_DEV_ST_multi_target_full_interval_coverage_pass"] = None

        timer.finalize()
        payload["B3_DEV_ST_multi_target_total_elapsed_seconds"] = _safe_float(
            payload.get("B3_DEV_timing_total_wall_elapsed_seconds")
        )
        ciss_total = payload.get("B3_DEV_ST_multi_target_CISS_reference_total_elapsed_seconds")
        st_total = payload.get("B3_DEV_ST_multi_target_total_elapsed_seconds")
        if (
            ciss_total is not None
            and st_total is not None
            and math.isfinite(float(ciss_total))
            and math.isfinite(float(st_total))
            and float(st_total) > 0.0
        ):
            payload["B3_DEV_ST_multi_target_speedup_vs_CISS"] = _safe_float(float(ciss_total) / float(st_total))

        if payload["B3_DEV_ST_multi_target_CISS_coverage_comparison_available"]:
            if payload["B3_DEV_ST_multi_target_full_interval_coverage_pass"]:
                verdict = "B3_DEV_ST_MULTI_TARGET_COVERAGE_PASS"
                rc = 0
            else:
                verdict = "B3_DEV_ST_MULTI_TARGET_COVERAGE_INCOMPLETE"
                payload["B3_DEV_failure_reason"] = (
                    f"missing_CISS={len(missing)};extra={len(extra)};"
                    f"union={len(union_freqs)};ciss_ref={len(ciss_ref_freqs)}"
                )
                rc = 2
        else:
            verdict = "B3_DEV_ST_MULTI_TARGET_SOLVE_PASS_CISS_REFERENCE_PENDING"
            rc = 0
    except Exception as exc:
        payload["B3_DEV_failure_reason"] = f"{type(exc).__name__}:{exc}"
        rc = 2
    finally:
        if "B3_DEV_timing_total_wall_elapsed_seconds" not in payload:
            timer.finalize()
        payload["next_step_verdict"] = verdict
        audit._write_json_atomic(out_json_path, payload)
        out_json_path.with_suffix(".md").write_text(
            f"# B3 dev ST multi-target coverage ({mesh_variant})\n\n"
            f"- verdict: `{verdict}`\n"
            f"- targets_hz: {targets_hz}\n"
            f"- union_accepted: {payload.get('B3_DEV_ST_multi_target_unique_accepted_frequency_count')}\n"
            f"- CISS_ref: {payload.get('B3_DEV_ST_multi_target_CISS_reference_frequency_count')}\n"
            f"- matches: {payload.get('B3_DEV_ST_multi_target_matches_CISS_count')}\n"
            f"- missing: {payload.get('B3_DEV_ST_multi_target_missing_CISS_frequencies')}\n"
            f"- coverage_pass: {payload.get('B3_DEV_ST_multi_target_full_interval_coverage_pass')}\n",
            encoding="utf-8",
        )
        audit._destroy_mats_deduped(mats)
    return rc


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
        dim_fail = _dev_record_active_dimension_target_range(payload, mesh_variant)
        if payload["B3_DEV_zero_row_column_cleanup_contract_pass"]:
            if not dim_fail:
                verdict = "B3_DEV_COARSE_OPERATOR_CONTRACT_PASS"
                rc = 0
            else:
                verdict = "B3_DEV_OPERATOR_CONTRACT_ACTIVE_DIMENSION_OUT_OF_TARGET"
                payload["B3_DEV_failure_reason"] = dim_fail
                rc = 2
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
        audit._write_json_atomic(_dev_out_json(_DEV_JSON_STEM_CONTRACT, mesh_variant), payload)
        audit._destroy_mats_deduped(mats)
    return rc


def _dev_lambda_close(value: Any, expected: float, *, rtol: float = 1.0e-9) -> bool:
    if value is None:
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(v):
        return False
    return abs(v - float(expected)) <= rtol * max(1.0, abs(float(expected)))


def _dev_st_targeting_requested() -> Tuple[float, float]:
    target_hz = float(audit.B3_CISS_VALIDATION_TARGET_HZ)
    return target_hz, float(audit._b3_hz_to_lambda_sq(target_hz))


def _dev_configure_coarse_krylovschur_sinvert_eps(
    eps: Any,
    A_active: Any,
    M_active: Any,
    *,
    payload: Dict[str, Any],
    target_lambda: float,
    target_hz: float,
) -> None:
    """KRYLOVSCHUR + ST.SINVERT with EPS target and ST shift at reference lambda."""
    from slepc4py import SLEPc

    payload["B3_DEV_ST_targeting_requested_frequency_hz"] = target_hz
    payload["B3_DEV_ST_targeting_requested_lambda"] = _safe_float(target_lambda)
    payload["B3_DEV_ST_solver_type_requested"] = "KRYLOVSCHUR"
    payload["B3_DEV_ST_problem_type_requested"] = "GNHEP"
    payload["B3_DEV_ST_ST_type_requested"] = "SINVERT"
    payload["B3_DEV_ST_which_requested"] = "TARGET_MAGNITUDE"
    payload["B3_DEV_ST_which_requested_raw"] = int(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    payload["B3_DEV_ST_setFromOptions_called"] = False

    eps.setOperators(A_active, M_active)
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.setTarget(float(target_lambda))
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
        raise RuntimeError(str(reason))
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


def _dev_eps_which_effective_fields(which_eff: Any) -> Dict[str, Any]:
    from slepc4py import SLEPc

    fields: Dict[str, Any] = {
        "B3_DEV_ST_which_effective_raw": None,
        "B3_DEV_ST_which_effective": None,
        "B3_DEV_ST_which_effective_normalized": None,
    }
    which_raw: Optional[int] = None
    if which_eff is not None:
        try:
            which_raw = int(which_eff)
        except (TypeError, ValueError):
            which_raw = None
    if which_raw is not None:
        fields["B3_DEV_ST_which_effective_raw"] = which_raw
        fields["B3_DEV_ST_which_effective"] = which_raw
        try:
            if which_raw == int(SLEPc.EPS.Which.TARGET_MAGNITUDE):
                fields["B3_DEV_ST_which_effective_normalized"] = "TARGET_MAGNITUDE"
        except Exception:
            pass
    if fields["B3_DEV_ST_which_effective_normalized"] is None:
        which_s = str(which_eff or "")
        if "target" in which_s.lower() and "magnitude" in which_s.lower():
            fields["B3_DEV_ST_which_effective_normalized"] = "TARGET_MAGNITUDE"
        elif which_s:
            fields["B3_DEV_ST_which_effective_normalized"] = which_s
    return fields


def _dev_introspect_st_targeting_after_setup(eps: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        out["B3_DEV_ST_target_effective"] = _safe_float(float(eps.getTarget()))
    except Exception:
        out["B3_DEV_ST_target_effective"] = None
    try:
        out.update(_dev_eps_which_effective_fields(eps.getWhichEigenpairs()))
    except Exception:
        out.update(_dev_eps_which_effective_fields(None))
    try:
        out["B3_DEV_ST_solver_type_effective"] = str(eps.getType())
    except Exception:
        out["B3_DEV_ST_solver_type_effective"] = None
    try:
        st = eps.getST()
        out["B3_DEV_ST_ST_type_effective"] = str(st.getType())
        shift_eff = None
        if hasattr(st, "getShift"):
            try:
                shift_eff = _safe_float(float(st.getShift()))
            except Exception:
                shift_eff = None
        out["B3_DEV_ST_shift_effective"] = shift_eff
    except Exception:
        out["B3_DEV_ST_ST_type_effective"] = None
        out["B3_DEV_ST_shift_effective"] = None
    return out


def _dev_evaluate_st_targeting_pass(payload: Dict[str, Any], target_lambda: float) -> None:
    from slepc4py import SLEPc

    st_type_eff = str(payload.get("B3_DEV_ST_ST_type_effective") or "")
    requested_which_value = int(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    which_raw = payload.get("B3_DEV_ST_which_effective_raw")
    which_target_pass = False
    if which_raw is not None:
        try:
            which_target_pass = int(which_raw) == requested_which_value
        except (TypeError, ValueError):
            which_target_pass = False
    if not which_target_pass:
        which_s = str(payload.get("B3_DEV_ST_which_effective_normalized") or payload.get("B3_DEV_ST_which_effective") or "")
        which_target_pass = bool("target" in which_s.lower() and "magnitude" in which_s.lower())
    payload["B3_DEV_ST_target_matches_requested_pass"] = bool(
        _dev_lambda_close(payload.get("B3_DEV_ST_target_effective"), target_lambda)
    )
    payload["B3_DEV_ST_shift_matches_requested_pass"] = bool(
        _dev_lambda_close(payload.get("B3_DEV_ST_shift_effective"), target_lambda)
    )
    payload["B3_DEV_ST_which_matches_requested_pass"] = bool(which_target_pass)
    payload["B3_DEV_ST_nearest_target_selection_effective_pass"] = bool(which_target_pass)
    payload["B3_DEV_ST_ST_sinvert_effective_pass"] = bool("sinvert" in st_type_eff.lower())
    payload["B3_DEV_ST_targeting_preflight_pass"] = bool(
        payload["B3_DEV_ST_target_matches_requested_pass"]
        and payload["B3_DEV_ST_shift_matches_requested_pass"]
        and payload["B3_DEV_ST_nearest_target_selection_effective_pass"]
        and payload["B3_DEV_ST_ST_sinvert_effective_pass"]
    )


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
        dim_fail = _dev_record_active_dimension_target_range(payload, mesh_variant)
        if dim_fail:
            payload["B3_DEV_failure_reason"] = dim_fail
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
        payload["B3_DEV_solve_elapsed_seconds"] = _safe_float(
            float(payload.get("B3_DEV_timing_eps_solve_end_elapsed_seconds", 0))
            - float(payload.get("B3_DEV_timing_eps_solve_begin_elapsed_seconds", 0))
        )
        payload["B3_DEV_CISS_solve_elapsed_seconds"] = payload["B3_DEV_solve_elapsed_seconds"]

        nconv, accepted = _dev_extract_modes_ciss(
            eps,
            A_active,
            built,
            payload,
            target_hz=target_hz,
            freq_lo=float(audit.B3_CISS_VALIDATION_FREQ_LO_HZ),
            freq_hi=float(audit.B3_CISS_VALIDATION_FREQ_HI_HZ),
        )
        payload["B3_DEV_CISS_converged_mode_count"] = int(nconv)
        payload["B3_DEV_setup_elapsed_seconds"] = payload.get("B3_DEV_CISS_setup_elapsed_seconds")
        if accepted:
            verdict = "B3_DEV_COARSE_CISS_BENCHMARK_PASS"
            rc = 0
        elif nconv > 0:
            verdict = "B3_DEV_COARSE_CISS_BENCHMARK_COMPLETED_NO_ACCEPTABLE_MODE"
            rc = 2
        else:
            verdict = "B3_DEV_COARSE_CISS_BENCHMARK_COMPLETED_NO_CONVERGED_MODE_IN_INTERVAL"
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
        audit._write_json_atomic(_dev_out_json(_DEV_JSON_STEM_CISS, mesh_variant), payload)
        audit._destroy_mats_deduped(mats)
    return rc


def _run_dev_coarse_st_targeting_preflight(pre: Dict[str, Any], mesh_variant: str) -> int:
    from slepc4py import SLEPc

    target_hz, target_lambda = _dev_st_targeting_requested()
    payload: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "B3_dev_coarse_krylovschur_sinvert_targeting_preflight_only",
        **_dev_mesh_bootstrap_payload(mesh_variant),
        "B3_DEV_ST_targeting_preflight_calls_setup": False,
        "B3_DEV_ST_targeting_preflight_calls_solve": False,
        "no_new_eigensolve_executed": True,
        "new_eigensolve_executed": False,
        "production_promotion": "BLOCKED",
    }
    timer = _B3DevTiming(payload)
    mats: List[Any] = []
    seen: set[int] = set()
    eps = None
    verdict = "B3_DEV_COARSE_ST_TARGETING_PREFLIGHT_BLOCKED"
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
        _dev_configure_coarse_krylovschur_sinvert_eps(
            eps,
            built["A_active"],
            built["M_active"],
            payload=payload,
            target_lambda=target_lambda,
            target_hz=target_hz,
        )
        timer.mark("eps_configure_end")

        timer.mark("eps_setup_begin")
        eps.setUp()
        payload["B3_DEV_ST_targeting_preflight_calls_setup"] = True
        timer.mark("eps_setup_end")
        payload.update(_dev_introspect_st_targeting_after_setup(eps))
        _dev_evaluate_st_targeting_pass(payload, target_lambda)
        if payload["B3_DEV_ST_targeting_preflight_pass"]:
            verdict = "B3_DEV_COARSE_ST_TARGETING_PREFLIGHT_PASS"
            rc = 0
        else:
            payload["B3_DEV_failure_reason"] = (
                f"target_match={payload.get('B3_DEV_ST_target_matches_requested_pass')};"
                f"shift_match={payload.get('B3_DEV_ST_shift_matches_requested_pass')};"
                f"which_match={payload.get('B3_DEV_ST_which_matches_requested_pass')};"
                f"st_sinvert={payload.get('B3_DEV_ST_ST_sinvert_effective_pass')}"
            )
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
        audit._write_json_atomic(_dev_out_json(_DEV_JSON_STEM_ST_TARGETING, mesh_variant), payload)
        _dev_out_md(_DEV_JSON_STEM_ST_TARGETING, mesh_variant).write_text(
            "# B3 dev coarse ST targeting preflight (no solve)\n\n"
            f"- verdict: `{verdict}`\n"
            f"- requested_hz: {payload.get('B3_DEV_ST_targeting_requested_frequency_hz')}\n"
            f"- target_effective: {payload.get('B3_DEV_ST_target_effective')}\n"
            f"- shift_effective: {payload.get('B3_DEV_ST_shift_effective')}\n"
            f"- which_effective: {payload.get('B3_DEV_ST_which_effective')}\n"
            f"- preflight_pass: {payload.get('B3_DEV_ST_targeting_preflight_pass')}\n",
            encoding="utf-8",
        )
        audit._destroy_mats_deduped(mats)
    return rc


def _run_dev_coarse_st_benchmark(pre: Dict[str, Any], mesh_variant: str) -> int:
    from slepc4py import SLEPc

    target_hz, target_lambda = _dev_st_targeting_requested()
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
        dim_fail = _dev_record_active_dimension_target_range(payload, mesh_variant)
        if dim_fail:
            payload["B3_DEV_failure_reason"] = dim_fail
            return 2

        timer.mark("eps_configure_begin")
        eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
        A_active = built["A_active"]
        M_active = built["M_active"]
        _dev_configure_coarse_krylovschur_sinvert_eps(
            eps,
            A_active,
            M_active,
            payload=payload,
            target_lambda=target_lambda,
            target_hz=target_hz,
        )
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
            payload,
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
        audit._write_json_atomic(_dev_out_json(_DEV_JSON_STEM_ST, mesh_variant), payload)
        audit._destroy_mats_deduped(mats)
    return rc


def _run_dev_dense_st_ciss_coverage_compare(mesh_variant: str) -> int:
    if mesh_variant != "L_dev_dense":
        print("[B3_DEV] dense coverage compare requires --B3-dev-mesh-variant L_dev_dense", flush=True)
        return 2
    freq_lo = float(audit.B3_CISS_VALIDATION_FREQ_LO_HZ)
    freq_hi = float(audit.B3_CISS_VALIDATION_FREQ_HI_HZ)
    st_path = _dev_out_json(_DEV_JSON_STEM_ST_MULTI_TARGET, mesh_variant)
    ciss_path = _dev_out_json(_DEV_JSON_STEM_CISS, mesh_variant)
    if not st_path.is_file():
        print(f"[B3_DEV] missing ST multi-target JSON: {st_path}", flush=True)
        return 2
    if not ciss_path.is_file():
        print(f"[B3_DEV] missing CISS JSON: {ciss_path}", flush=True)
        return 2
    st_data = json.loads(st_path.read_text(encoding="utf-8"))
    ciss_loaded, ciss_freqs, ciss_data = _dev_load_ciss_reference_accepted_frequencies(
        mesh_variant, freq_lo=freq_lo, freq_hi=freq_hi
    )
    if not ciss_loaded or not ciss_freqs:
        print("[B3_DEV] CISS reference frequencies unavailable", flush=True)
        return 2
    union_freqs = list(st_data.get("B3_DEV_ST_multi_target_unique_accepted_frequencies") or [])
    matches, missing, extra = _dev_compare_frequency_sets(
        union_freqs, ciss_freqs, match_tol_hz=B3_DEV_ST_MULTI_CISS_MATCH_TOL_HZ
    )
    coverage_pass = bool(len(missing) == 0 and matches == len(ciss_freqs))
    ciss_total = (ciss_data or {}).get("B3_DEV_CISS_total_elapsed_seconds") or (
        ciss_data or {}
    ).get("B3_DEV_timing_total_wall_elapsed_seconds")
    st_total = st_data.get("B3_DEV_ST_multi_target_total_elapsed_seconds")
    speedup = None
    if (
        ciss_total is not None
        and st_total is not None
        and math.isfinite(float(ciss_total))
        and math.isfinite(float(st_total))
        and float(st_total) > 0.0
    ):
        speedup = float(ciss_total) / float(st_total)
    summary = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "B3_dev_dense_st_ciss_coverage_compare_only",
        "B3_DEV_mesh_variant": mesh_variant,
        "B3_DEV_ST_multi_target_unique_accepted_frequencies": union_freqs,
        "B3_DEV_ST_multi_target_unique_accepted_frequency_count": len(union_freqs),
        "B3_DEV_ST_multi_target_CISS_reference_frequencies": ciss_freqs,
        "B3_DEV_ST_multi_target_CISS_reference_frequency_count": len(ciss_freqs),
        "B3_DEV_ST_multi_target_matches_CISS_count": int(matches),
        "B3_DEV_ST_multi_target_missing_CISS_frequencies": missing,
        "B3_DEV_ST_multi_target_extra_frequencies_in_interval": extra,
        "B3_DEV_ST_multi_target_full_interval_coverage_pass": coverage_pass,
        "B3_DEV_ST_multi_target_CISS_reference_total_elapsed_seconds": _safe_float(ciss_total),
        "B3_DEV_ST_multi_target_total_elapsed_seconds": _safe_float(st_total),
        "B3_DEV_ST_multi_target_speedup_vs_CISS": _safe_float(speedup),
        "next_step_verdict": (
            "B3_DEV_DENSE_ST_CISS_COVERAGE_PASS" if coverage_pass else "B3_DEV_DENSE_ST_CISS_COVERAGE_INCOMPLETE"
        ),
    }
    out_path = audit.CONV_DIAG / "v2_B3_DEV_dense_st_ciss_coverage_compare.json"
    audit._write_json_atomic(out_path, summary)
    print(f"[B3_DEV] dense_coverage_compare_written={out_path}", flush=True)
    return 0 if coverage_pass else 2


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

    for variant in sorted(B3_DEV_ALLOWED_MESH_VARIANTS):
        _load_row(_dev_out_json(_DEV_JSON_STEM_CONTRACT, variant), f"operator_contract:{variant}")
        _load_row(_dev_out_json(_DEV_JSON_STEM_CISS, variant), f"CISS-DIRECT-STABLE:{variant}")
        _load_row(_dev_out_json(_DEV_JSON_STEM_ST, variant), f"KRYLOVSCHUR-ST-SINVERT-MUMPS:{variant}")
        st_mt = _dev_out_json(_DEV_JSON_STEM_ST_MULTI_TARGET, variant)
        if st_mt.is_file():
            data = json.loads(st_mt.read_text(encoding="utf-8"))
            rows.append(
                {
                    "solver": "KRYLOVSCHUR-ST-MULTI-TARGET",
                    "mesh_variant": data.get("B3_DEV_mesh_variant", variant),
                    "active_dimension": data.get("B3_DEV_active_dimension"),
                    "operator_build_time_s": data.get("B3_DEV_timing_A_active_M_active_ready_elapsed_seconds"),
                    "setup_time_s": data.get("B3_DEV_ST_multi_target_total_setup_elapsed_seconds"),
                    "solve_time_s": data.get("B3_DEV_ST_multi_target_total_solve_elapsed_seconds"),
                    "total_time_s": data.get("B3_DEV_ST_multi_target_total_elapsed_seconds"),
                    "returned_frequencies_hz": data.get("B3_DEV_ST_multi_target_unique_accepted_frequencies"),
                    "closest_to_244_39_hz": None,
                    "closest_distance_hz": None,
                    "accepted_mode_count": data.get("B3_DEV_ST_multi_target_unique_accepted_frequency_count"),
                    "peak_memory_rss_mb": data.get("B3_DEV_peak_memory_rss_mb"),
                    "verdict": data.get("next_step_verdict"),
                }
            )

    summary = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "B3_dev_solver_benchmark_compare_only",
        "B3_DEV_mesh_variants_compared": sorted(B3_DEV_ALLOWED_MESH_VARIANTS),
        "B3_DEV_mesh_is_solver_smoke_test_only": True,
        "B3_DEV_mesh_not_authorized_for_final_physics_validation": True,
        "comparison_rows": rows,
        "no_automatic_production_winner_selected": True,
    }
    summary_path = audit.CONV_DIAG / f"{_DEV_JSON_STEM_SUMMARY}.json"
    summary_md = audit.CONV_DIAG / f"{_DEV_JSON_STEM_SUMMARY}.md"
    audit._write_json_atomic(summary_path, summary)
    summary_md.write_text(
        "# B3 dev solver benchmark summary (coarse vs refined)\n\n"
        + "\n".join(
            f"- **{r['solver']}** [{r.get('mesh_variant')}]: active_dim={r.get('active_dimension')} "
            f"build={r.get('operator_build_time_s')}s setup={r.get('setup_time_s')}s "
            f"solve={r.get('solve_time_s')}s total={r.get('total_time_s')}s "
            f"accepted={r.get('accepted_mode_count')} verdict={r.get('verdict')}"
            for r in rows
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[B3_DEV] summary_written={summary_path}", flush=True)
    return 0


def is_b3_dev_mode(argv: List[str]) -> bool:
    dev_flags = (
        B3_DEV_COARSE_CONTRACT_ARG,
        B3_DEV_COARSE_CISS_ARG,
        B3_DEV_COARSE_ST_ARG,
        B3_DEV_COARSE_ST_TARGETING_PREFLIGHT_ARG,
        B3_DEV_REFINED_ST_MULTI_TARGET_ARG,
        B3_DEV_DENSE_COVERAGE_COMPARE_ARG,
        B3_DEV_COMPARE_ARG,
    )
    return any(f in argv for f in dev_flags)


def run_b3_dev_mode(argv: List[str], pre: Dict[str, Any]) -> int:
    mesh_variant = _parse_dev_mesh_variant(argv)
    if mesh_variant not in B3_DEV_ALLOWED_MESH_VARIANTS:
        print(
            f"[B3_DEV] unsupported mesh_variant={mesh_variant} "
            f"allowed={sorted(B3_DEV_ALLOWED_MESH_VARIANTS)}",
            flush=True,
        )
        return 2
    if B3_DEV_COARSE_CONTRACT_ARG in argv:
        return _run_dev_coarse_contract(pre, mesh_variant)
    if B3_DEV_COARSE_CISS_ARG in argv:
        return _run_dev_coarse_ciss_benchmark(pre, mesh_variant)
    if B3_DEV_COARSE_ST_TARGETING_PREFLIGHT_ARG in argv:
        return _run_dev_coarse_st_targeting_preflight(pre, mesh_variant)
    if B3_DEV_COARSE_ST_ARG in argv:
        return _run_dev_coarse_st_benchmark(pre, mesh_variant)
    if B3_DEV_REFINED_ST_MULTI_TARGET_ARG in argv:
        try:
            st_targets_hz = _parse_dev_st_multi_targets_hz(argv)
        except ValueError as exc:
            print(f"[B3_DEV] {exc}", flush=True)
            return 2
        return _run_dev_st_multi_target_coverage_benchmark(
            pre, mesh_variant, targets_hz=st_targets_hz
        )
    if B3_DEV_DENSE_COVERAGE_COMPARE_ARG in argv:
        return _run_dev_dense_st_ciss_coverage_compare(mesh_variant)
    if B3_DEV_COMPARE_ARG in argv:
        return _run_dev_compare_summary(mesh_variant)
    print("[B3_DEV] no dev mode flag recognized", flush=True)
    return 2
