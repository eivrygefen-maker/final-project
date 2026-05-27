#!/usr/bin/env python3
"""
Report-only: certify empirical M_uu null basis from lossless EPS candidates and projection preflight.

Reads existing lossless vectors + reassembled replay operators only. Does not call eps.solve().
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from mpi4py import MPI

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
for _p in (SCRIPT_DIR, REPO_ROOT / "FEM" / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from v2_clean_adjudication_lane import OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1
from v2_mesh_convergence_common import (
    CONV_DIAG,
    load_manifest,
    mesh_path,
    sample_spec_from_case,
    solve_case_dir,
    write_json,
)
from v2_build_coupled_acoustic_seed import _assemble_reduced_coupled_replay, _extract_layout_maps
from v2_unreg_offset_report_evaluator import load_seed_with_diagnostics

CASE_ID = "baseline_coupled_v2"

OUT_JSON = (
    CONV_DIAG
    / "v2_lossless_adjudication_v1_Muu_null_basis_certification_and_projection_preflight.json"
)
OUT_MD = (
    CONV_DIAG
    / "v2_lossless_adjudication_v1_Muu_null_basis_certification_and_projection_preflight.md"
)
MASS_RANK_AUDIT_JSON = (
    CONV_DIAG / "v2_lossless_adjudication_v1_u_mass_rank_and_disjoint_partition_audit.json"
)

RANK_PROBE_TOL = 1e-10
SVD_RCOND = 1e-10
CERTIFIED_NULL_REL_TOL = 1e-12
SEED_PRESERVE_REL_TOL = 1e-10
SUPPRESSION_REMOVED_MEDIAN_TOL = 0.99
SUPPRESSION_REMOVED_MIN_TOL = 0.90
SUPPRESSION_FRACTION_ABOVE_0_99 = 0.90
DUPLICATE_COSINE_TOL = 0.999

PRIMARY_BLOCKER = "LOSSLESS_ST_RETURNED_U_SHELL_MASS_MATRIX_KERNEL_MODES"
STRATEGY_PROJECTED_V1 = "MAPPING_FIXED_UNREGULARIZED_LOSSLESS_NULLSPACE_PROJECTED_ADJUDICATION_V1"
STRATEGY_INSUFFICIENT = "CERTIFIED_NULL_SUBSPACE_INSUFFICIENT_TO_REMOVE_RETURNED_FAMILY"
STRATEGY_NOT_AUTHORIZED = "NOT_AUTHORIZED_PENDING_PREFLIGHT_GATES"


def _coalesce_list(*candidates: Any) -> list:
    for c in candidates:
        if isinstance(c, list):
            return c
    return []


def _atomic_load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _np_int32_1d(raw: Any) -> np.ndarray:
    if raw is None:
        return np.asarray([], dtype=np.int32)
    if hasattr(raw, "array") and not isinstance(raw, (list, tuple, np.ndarray)):
        try:
            raw = raw.array
        except Exception:
            pass
    return np.asarray(raw, dtype=np.int32).ravel()


def _petsc_matvec_u(op: Any, x_u: np.ndarray, u_idx: np.ndarray) -> np.ndarray:
    from physical_fsi_seed_residual_audit import _petsc_matvec, _petsc_vec_from_array

    n = int(op.getSize()[1])
    x_full = np.zeros(n, dtype=np.float64)
    x_full[np.asarray(u_idx, dtype=np.int32)] = np.asarray(x_u, dtype=np.float64).ravel()
    vx = _petsc_vec_from_array(op, x_full)
    try:
        y, vy = _petsc_matvec(op, vx)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
    finally:
        vx.destroy()
        if vy is not None:
            vy.destroy()
    return y_arr[np.asarray(u_idx, dtype=np.int32)]


def _project_u(x_u: np.ndarray, Q: np.ndarray) -> np.ndarray:
    if Q.size == 0:
        return np.asarray(x_u, dtype=np.float64).ravel()
    return np.asarray(x_u, dtype=np.float64).ravel() - Q @ (Q.T @ x_u)


def _project_full(vec: np.ndarray, Q: np.ndarray, u_idx: np.ndarray) -> np.ndarray:
    out = np.asarray(vec, dtype=np.float64).ravel().copy()
    out[u_idx] = _project_u(out[u_idx], Q)
    return out


def _orthonormalize_columns(Q_raw: np.ndarray) -> np.ndarray:
    if Q_raw.size == 0:
        return np.zeros((Q_raw.shape[0], 0), dtype=np.float64)
    Q_orth, _ = np.linalg.qr(np.asarray(Q_raw, dtype=np.float64), mode="reduced")
    return Q_orth.astype(np.float64, copy=False)


def _resolve_eps_slot(mode_row: Dict[str, Any], bank_by_lossless_rel: Dict[str, Dict[str, Any]]) -> int:
    for key in ("eps_slot_index", "candidate_index", "mode_index"):
        val = mode_row.get(key)
        if val is not None:
            try:
                slot = int(val)
                if slot >= 0:
                    return slot
            except (TypeError, ValueError):
                pass
    rel = mode_row.get("vector_file_lossless")
    if rel:
        bank_row = bank_by_lossless_rel.get(str(rel).replace("\\", "/"))
        if bank_row:
            for key in ("eps_slot_index", "candidate_index"):
                val = bank_row.get(key)
                if val is not None:
                    try:
                        slot = int(val)
                        if slot >= 0:
                            return slot
                    except (TypeError, ValueError):
                        pass
    return -1


def _audit_random_probe_interpretation(mass_rank: Dict[str, Any]) -> Dict[str, Any]:
    probe = mass_rank.get("M_uu_nullity_probe") or {}
    activity = mass_rank.get("structural_M_uu_activity") or {}
    summary = mass_rank.get("M_uu_nullity_or_rank_summary", "")
    null_ub = int(probe.get("estimated_nullity_dimension_upper_bound", 0))
    return {
        "prior_report_summary": summary,
        "probe_method": (
            "Independent standard-normal directions z in R^{n_u_active}, normalized, "
            "then ||M_uu z|| recorded (same backend as mass-rank audit)."
        ),
        "probe_dimension": int(probe.get("probe_dim", 0)),
        "probe_tolerance": RANK_PROBE_TOL,
        "random_probe_detected_null_vectors": bool(null_ub > 0),
        "empirical_eps_null_vectors_detected": True,
        "global_Muu_nullity_not_certified": True,
        "nullity_ub_equals_zero_means": (
            f"No random probe landed in kernel at {RANK_PROBE_TOL:g}; "
            "does NOT certify global nullity(M_uu)=0."
        ),
        "M_uu_nonzero_row_count": activity.get("M_uu_nonzero_row_count"),
        "M_uu_nonzero_column_count": activity.get("M_uu_nonzero_column_count"),
    }


def _duplicate_clusters(X: np.ndarray, tol: float = DUPLICATE_COSINE_TOL) -> Dict[str, Any]:
    k = int(X.shape[1])
    if k == 0:
        return {"cluster_count": 0, "pairs_above_tol": 0}
    sim = np.abs(X.T @ X)
    pairs = int(np.sum(np.triu(sim > tol, k=1)))
    parent = list(range(k))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(k):
        for j in range(i + 1, k):
            if sim[i, j] > tol:
                union(i, j)
    roots = {find(i) for i in range(k)}
    return {
        "cluster_count": len(roots),
        "pairs_above_tol": pairs,
        "max_pairwise_cosine": float(np.max(sim[np.triu_indices(k, k=1)]) if k > 1 else 1.0),
    }


def _shell_fractions(x_u: np.ndarray, shell_pos: np.ndarray, non_shell_pos: np.ndarray) -> Dict[str, float]:
    n = float(np.linalg.norm(x_u))
    if n <= 0:
        return {"shell_l2_fraction": 0.0, "non_shell_l2_fraction": 0.0}
    shell = float(np.linalg.norm(x_u[shell_pos])) if shell_pos.size else 0.0
    nons = float(np.linalg.norm(x_u[non_shell_pos])) if non_shell_pos.size else 0.0
    return {"shell_l2_fraction": shell / n, "non_shell_l2_fraction": nons / n}


def _candidate_projection_stats(
    *,
    modes: List[Dict[str, Any]],
    lossless_path_fn,
    load_vec_fn,
    u_to_W: np.ndarray,
    Q: np.ndarray,
    bank_by_lossless_rel: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    per_slot: List[Dict[str, Any]] = []
    removed_fracs: List[float] = []
    residual_fracs: List[float] = []
    for m in modes:
        lp = lossless_path_fn(m)
        if lp is None:
            continue
        vec = np.asarray(load_vec_fn(lp), dtype=np.float64).ravel()
        x_u = vec[u_to_W]
        xn = float(np.linalg.norm(x_u))
        if xn <= 0:
            continue
        x_u_proj = _project_u(x_u, Q)
        rn = float(np.linalg.norm(x_u_proj))
        removed = float(np.linalg.norm(x_u - x_u_proj) / xn)
        residual_frac = float(rn / xn)
        removed_fracs.append(removed)
        residual_fracs.append(residual_frac)
        slot = _resolve_eps_slot(m, bank_by_lossless_rel)
        per_slot.append(
            {
                "eps_slot_index": slot,
                "vector_file_lossless": str(m.get("vector_file_lossless", "")).replace("\\", "/"),
                "removed_norm_fraction": removed,
                "residual_norm_fraction_after_projection": residual_frac,
                "residual_norm_after_projection": rn,
            }
        )

    def _counts(thr: float) -> int:
        return int(sum(1 for r in removed_fracs if r >= thr))

    summary = {
        "count": len(per_slot),
        "removed_norm_fraction": {
            "min": float(np.min(removed_fracs)) if removed_fracs else 0.0,
            "median": float(np.median(removed_fracs)) if removed_fracs else 0.0,
            "max": float(np.max(removed_fracs)) if removed_fracs else 0.0,
        },
        "residual_norm_fraction_after_projection": {
            "min": float(np.min(residual_fracs)) if residual_fracs else 0.0,
            "median": float(np.median(residual_fracs)) if residual_fracs else 0.0,
            "max": float(np.max(residual_fracs)) if residual_fracs else 0.0,
        },
        "count_candidates_removed_above_0_99": _counts(0.99),
        "count_candidates_removed_above_0_999": _counts(0.999),
        "count_candidates_removed_above_0_999999": _counts(0.999999),
    }
    return per_slot, summary


def main() -> int:
    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[null_basis_preflight] Requires mpiexec -n 1", file=sys.stderr)
        return 2

    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
    case_dir = solve_case_dir("L_mid", CASE_ID)
    out_dir = case_dir / OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1
    mesh_file = mesh_path("L_mid", CASE_ID)
    seed_npy = case_dir / "diagnostics" / "acoustic_coupled_seed.npy"

    mass_rank = _atomic_load_json(MASS_RANK_AUDIT_JSON)
    eps_auth = _atomic_load_json(CONV_DIAG / "v2_lossless_adjudication_v1_eps_authorization_record.json")
    eps_run_count = int(eps_auth.get("eps_run_count_for_this_lane", 0) or 0)

    sample = sample_spec_from_case(case)
    A, M, cfg = _assemble_reduced_coupled_replay(mesh_file, sample, coupling_enabled=True)
    maps = _extract_layout_maps(cfg, A)
    u_to_W = _np_int32_1d(maps.get("u_to_W"))
    p_to_W = _np_int32_1d(maps.get("p_to_W"))

    try:
        from run_v2_lossless_adjudication_v1_u_mass_rank_and_disjoint_partition_audit import (
            _build_disjoint_u_partition,
        )
        from run_v2_lossless_adjudication_v1_u_active_nullspace_attribution import (
            _build_tag_subsets_in_reduced_u,
            _matvec_norms as _full_matvec_norms,
        )
        from fem_mode_array_utils import load_mode_dense_f64_lossless

        operator_size = int(A.getSize()[0])
        tag_map = _build_tag_subsets_in_reduced_u(
            mesh_file=mesh_file,
            sample=sample,
            u_to_W=u_to_W,
            p_to_W=p_to_W,
            operator_size=operator_size,
        )
        partition, _ = _build_disjoint_u_partition(tag_map["subsets"], u_to_W)
        shell_global = set(int(i) for i in partition["u_shell_physical_union_excluding_tag5"].tolist())
        shell_pos = np.asarray(
            [i for i, g in enumerate(u_to_W.tolist()) if int(g) in shell_global], dtype=np.int32
        )
        non_shell_pos = np.asarray(
            [i for i, g in enumerate(u_to_W.tolist()) if int(g) not in shell_global], dtype=np.int32
        )

        modes = _coalesce_list(_atomic_load_json(out_dir / "diagnostics/mode_energy_summary.json").get("modes"))
        bank = _atomic_load_json(out_dir / "diagnostics/eps_candidate_bank.json")
        bank_rows = _coalesce_list(bank.get("saved_mode_rows"), bank.get("candidates"))
        bank_by_lossless_rel: Dict[str, Dict[str, Any]] = {}
        for row in bank_rows:
            if not isinstance(row, dict):
                continue
            rel = row.get("vector_file_lossless")
            if rel:
                bank_by_lossless_rel[str(rel).replace("\\", "/")] = row

        def lossless_path(mode_row: Dict[str, Any]) -> Optional[Path]:
            rel = mode_row.get("vector_file_lossless")
            if not rel:
                return None
            p = out_dir / str(rel)
            return p if p.is_file() else None

        columns: List[np.ndarray] = []
        for m in modes:
            lp = lossless_path(m)
            if lp is None:
                continue
            vec = np.asarray(load_mode_dense_f64_lossless(lp), dtype=np.float64).ravel()
            x_u = vec[u_to_W]
            xn = float(np.linalg.norm(x_u))
            if xn <= 0:
                continue
            columns.append((x_u / xn).astype(np.float64, copy=False))

        n_u = int(u_to_W.size)
        n_cand = len(columns)
        if n_cand == 0:
            raise RuntimeError("no lossless candidate u_active vectors loaded")

        X = np.column_stack(columns)
        gram_sv = np.linalg.svd(X.T @ X, compute_uv=False)
        X_svd_u, s_vals, _ = np.linalg.svd(X, full_matrices=False)
        rank_full = int(np.sum(s_vals > SVD_RCOND * float(s_vals[0]))) if s_vals.size else 0
        Q_full = X_svd_u[:, :rank_full].astype(np.float64, copy=False)

        independent_directions: List[Dict[str, Any]] = []
        certified_indices: List[int] = []
        for i in range(rank_full):
            q = Q_full[:, i]
            qn = float(np.linalg.norm(q))
            Mq = _petsc_matvec_u(M, q, u_to_W)
            Aq = _petsc_matvec_u(A, q, u_to_W)
            rel_m = float(np.linalg.norm(Mq) / max(qn, 1e-300))
            is_null = bool(rel_m < CERTIFIED_NULL_REL_TOL)
            if is_null:
                certified_indices.append(i)
            fr = _shell_fractions(q, shell_pos, non_shell_pos)
            independent_directions.append(
                {
                    "basis_index": i,
                    "singular_value": float(s_vals[i]),
                    "relative_mass_action": rel_m,
                    "mass_null_direction": is_null,
                    "l2_norm_Muu_q": float(np.linalg.norm(Mq)),
                    "qH_Muu_q": float(np.vdot(q, Mq).real),
                    **fr,
                }
            )

        Q_certified_raw = Q_full[:, certified_indices] if certified_indices else np.zeros((n_u, 0))
        Q_certified_null = _orthonormalize_columns(Q_certified_raw)
        certified_dim = int(Q_certified_null.shape[1])
        certified_null_basis_certified = bool(
            certified_dim > 0
            and all(independent_directions[i]["mass_null_direction"] for i in certified_indices)
        )

        probe_audit = _audit_random_probe_interpretation(mass_rank)

        seed_info = load_seed_with_diagnostics(seed_npy)
        seed_vec = np.asarray(seed_info["seed_array"], dtype=np.float64).ravel()
        seed_before = _full_matvec_norms(A, M, seed_vec, u_idx=u_to_W, p_idx=p_to_W)

        seed_proj_cert = _project_full(seed_vec, Q_certified_null, u_to_W)
        seed_after_cert = _full_matvec_norms(A, M, seed_proj_cert, u_idx=u_to_W, p_idx=p_to_W)
        seed_rel_change_cert = float(
            np.linalg.norm(seed_proj_cert - seed_vec) / max(float(np.linalg.norm(seed_vec)), 1e-300)
        )
        seed_preservation_cert = bool(seed_rel_change_cert <= SEED_PRESERVE_REL_TOL)

        seed_proj_full = _project_full(seed_vec, Q_full, u_to_W)
        seed_rel_change_full = float(
            np.linalg.norm(seed_proj_full - seed_vec) / max(float(np.linalg.norm(seed_vec)), 1e-300)
        )

        cand_cert, cand_cert_summary = _candidate_projection_stats(
            modes=modes,
            lossless_path_fn=lossless_path,
            load_vec_fn=load_mode_dense_f64_lossless,
            u_to_W=u_to_W,
            Q=Q_certified_null,
            bank_by_lossless_rel=bank_by_lossless_rel,
        )
        cand_full, cand_full_summary = _candidate_projection_stats(
            modes=modes,
            lossless_path_fn=lossless_path,
            load_vec_fn=load_mode_dense_f64_lossless,
            u_to_W=u_to_W,
            Q=Q_full,
            bank_by_lossless_rel=bank_by_lossless_rel,
        )

        removed_med_cert = float(cand_cert_summary["removed_norm_fraction"]["median"])
        removed_min_cert = float(cand_cert_summary["removed_norm_fraction"]["min"])
        n_cand_proj = int(cand_cert_summary["count"])
        frac_above_099 = (
            float(cand_cert_summary["count_candidates_removed_above_0_99"]) / max(n_cand_proj, 1)
        )

        family_suppressed_cert = bool(
            n_cand_proj > 0
            and removed_med_cert >= SUPPRESSION_REMOVED_MEDIAN_TOL
            and removed_min_cert >= SUPPRESSION_REMOVED_MIN_TOL
            and frac_above_099 >= SUPPRESSION_FRACTION_ABOVE_0_99
        )

        authorization_gates_pass = bool(
            certified_null_basis_certified
            and seed_preservation_cert
            and family_suppressed_cert
        )

        if authorization_gates_pass:
            recommended_future_strategy = STRATEGY_PROJECTED_V1
        elif certified_null_basis_certified and seed_preservation_cert and not family_suppressed_cert:
            recommended_future_strategy = STRATEGY_INSUFFICIENT
        else:
            recommended_future_strategy = STRATEGY_NOT_AUTHORIZED

        report: Dict[str, Any] = {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evidence_scope": "report_only_no_eps",
            "primary_blocker": PRIMARY_BLOCKER,
            "no_new_eigensolve_executed": True,
            "additional_eps_authorized": False,
            "certified_empirical_null_basis_dimension": certified_dim,
            "certified_null_basis_indices": certified_indices,
            "certified_null_basis_threshold": CERTIFIED_NULL_REL_TOL,
            "certified_null_basis_certified": certified_null_basis_certified,
            "numerical_rank_Q_full_dimension": rank_full,
            "empirical_null_basis_dimension_in_returned_set": rank_full,
            "empirical_null_basis_certified": False,
            "seed_projection_preservation_pass_certified_null": seed_preservation_cert,
            "seed_relative_change_norm_ratio_certified_null": seed_rel_change_cert,
            "candidate_removed_norm_fraction_certified_null": cand_cert_summary["removed_norm_fraction"],
            "candidate_residual_norm_fraction_after_certified_null_projection": cand_cert_summary[
                "residual_norm_fraction_after_projection"
            ],
            "count_candidates_removed_above_0_99": cand_cert_summary["count_candidates_removed_above_0_99"],
            "count_candidates_removed_above_0_999": cand_cert_summary["count_candidates_removed_above_0_999"],
            "count_candidates_removed_above_0_999999": cand_cert_summary["count_candidates_removed_above_0_999999"],
            "projected_existing_mass_null_family_sufficiently_suppressed_by_certified_null_basis": family_suppressed_cert,
            "recommended_future_strategy": recommended_future_strategy,
            "authorization_gates_pass": authorization_gates_pass,
            "single_run_guard_audit": {
                "eps_run_count_for_this_lane": eps_run_count,
                "no_additional_eps_run_authorized": True,
                "re_invoking_authorized_runner_would_block_eps": bool(eps_run_count >= 1),
            },
            "random_probe_interpretation_audit": probe_audit,
            "empirical_null_basis": {
                "candidate_vector_count": n_cand,
                "n_u_active": n_u,
                "gram_singular_values": gram_sv.tolist(),
                "candidate_matrix_singular_values": s_vals.tolist(),
                "numerical_rank_X": rank_full,
                "duplicate_cluster_audit": _duplicate_clusters(X),
                "independent_direction_diagnostics": independent_directions,
            },
            "projectors": {
                "Q_full": {
                    "purpose": "diagnostic_comparison_only_never_authorizing",
                    "dimension": rank_full,
                    "Q_shape": [n_u, rank_full],
                    "seed_relative_change_norm_ratio": seed_rel_change_full,
                    "candidate_projection_summary": cand_full_summary,
                    "projected_existing_mass_null_family_removed": bool(
                        cand_full_summary["removed_norm_fraction"]["median"] >= SUPPRESSION_REMOVED_MEDIAN_TOL
                    ),
                    "authorization_eligible": False,
                },
                "Q_certified_null": {
                    "purpose": "only_projector_eligible_for_future_eps_authorization",
                    "dimension": certified_dim,
                    "Q_shape": [n_u, certified_dim],
                    "certified_null_basis_indices": certified_indices,
                    "certified_null_basis_threshold": CERTIFIED_NULL_REL_TOL,
                    "seed_before_projection": seed_before,
                    "seed_after_projection": seed_after_cert,
                    "seed_projection_preservation_pass": seed_preservation_cert,
                    "seed_relative_change_norm_ratio": seed_rel_change_cert,
                    "candidate_per_slot": cand_cert,
                    "candidate_projection_summary": cand_cert_summary,
                    "suppression_threshold_justification": {
                        "median_removed_norm_fraction_min": SUPPRESSION_REMOVED_MEDIAN_TOL,
                        "min_removed_norm_fraction_min": SUPPRESSION_REMOVED_MIN_TOL,
                        "fraction_of_candidates_removed_above_0_99_min": SUPPRESSION_FRACTION_ABOVE_0_99,
                        "rationale": (
                            "Returned EPS modes are u_active-dominated mass-null vectors. "
                            "Certified-null projection must remove the dominant offending component "
                            "from most of the returned family without using non-certified directions "
                            "that have measurable ||M_uu q||."
                        ),
                    },
                    "projected_existing_mass_null_family_sufficiently_suppressed": family_suppressed_cert,
                    "authorization_eligible": authorization_gates_pass,
                },
            },
            "future_diagnostic_strategy_design": {
                "strategy_name": STRATEGY_PROJECTED_V1,
                "authorized_to_execute": False,
                "preflight_gates_pass": authorization_gates_pass,
                "use_projector": "Q_certified_null_only",
                "do_not_use_projector": "Q_full",
                "must_remain_identical": [
                    "V2 physics / forms / BCs / scaling",
                    "true seed",
                    "sigma target and ladder",
                    "ST=SINVERT",
                    "unregularized policy",
                    "corrected eigenvalue mapping",
                    "lossless authoritative persistence",
                    "physical replay gates",
                ],
            },
            "root_cause_status_refresh": {
                "primary_blocker": PRIMARY_BLOCKER,
                "serialization_ruled_out_as_active_cause": True,
                "single_lossless_eps_run_consumed": True,
                "v2_physical_model_invalidated": False,
                "st_viability": (
                    "projected_eps_ready_for_authorization_review"
                    if authorization_gates_pass
                    else "unresolved_pending_certified_null_projection_preflight"
                ),
                "additional_eps": "NOT_AUTHORIZED",
            },
        }

        write_json(OUT_JSON, report)

        md = [
            "# M_uu null-basis certification and projection preflight",
            "",
            f"Generated: {report['generated_utc']}",
            "",
            f"**primary_blocker:** `{PRIMARY_BLOCKER}`",
            f"certified_empirical_null_basis_dimension={certified_dim}",
            f"certified_null_basis_certified={certified_null_basis_certified}",
            f"seed_projection_preservation_pass_certified_null={seed_preservation_cert}",
            f"candidate_removed_norm_fraction_certified_null_median={removed_med_cert}",
            f"projected_existing_mass_null_family_sufficiently_suppressed_by_certified_null_basis={family_suppressed_cert}",
            f"recommended_future_strategy={recommended_future_strategy}",
            "",
            f"Q_full (diagnostic only, dim={rank_full}): median_removed="
            f"{cand_full_summary['removed_norm_fraction']['median']}",
            "",
        ]
        OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

        try:
            from write_v2_st_singular_mass_rehabilitation_plan import main as rehab_main
            from run_v2_solver_root_cause_and_forward_risk_audit import main as audit_main

            rehab_main()
            audit_main()
        except Exception as exc:
            print(f"[null_basis_preflight] status_refresh_warning={type(exc).__name__}:{exc}", flush=True)

        print(f"[null_basis_preflight] certified_empirical_null_basis_dimension={certified_dim}", flush=True)
        print(f"[null_basis_preflight] certified_null_basis_certified={certified_null_basis_certified}", flush=True)
        print(
            f"[null_basis_preflight] seed_projection_preservation_pass_certified_null={seed_preservation_cert}",
            flush=True,
        )
        print(
            f"[null_basis_preflight] candidate_removed_norm_fraction_certified_null_median={removed_med_cert}",
            flush=True,
        )
        print(
            "[null_basis_preflight] "
            f"projected_existing_mass_null_family_sufficiently_suppressed_by_certified_null_basis={family_suppressed_cert}",
            flush=True,
        )
        print(f"[null_basis_preflight] recommended_future_strategy={recommended_future_strategy}", flush=True)
        print("[null_basis_preflight] no_new_eigensolve_executed=True", flush=True)
        print("[null_basis_preflight] additional_eps=NOT_AUTHORIZED", flush=True)
        return 0
    finally:
        try:
            A.destroy()
            M.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
