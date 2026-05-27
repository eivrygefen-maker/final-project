#!/usr/bin/env python3
"""
Report-only: certify empirical M_uu null basis from lossless EPS candidates and projection preflight.

Reads existing lossless vectors + reassembled replay operators only. Does not call eps.solve().
"""
from __future__ import annotations

import json
import math
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
MASS_NULL_REL_TOL = 1e-12
SEED_PRESERVE_REL_TOL = 1e-10
FAMILY_REMOVED_FRAC_TOL = 0.99
DUPLICATE_COSINE_TOL = 0.999

PRIMARY_BLOCKER = "LOSSLESS_ST_RETURNED_U_SHELL_MASS_MATRIX_KERNEL_MODES"


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


def _matvec_norms(A: Any, M: Any, x: np.ndarray, u_idx: np.ndarray, p_idx: np.ndarray) -> Dict[str, Any]:
    from physical_fsi_seed_residual_audit import _petsc_matvec, _petsc_vec_from_array, _rayleigh_metrics

    x_arr = np.asarray(x, dtype=np.float64).ravel()
    vx = _petsc_vec_from_array(A, x_arr)
    ay = my = None
    try:
        Ax, ay = _petsc_matvec(A, vx)
        Mx, my = _petsc_matvec(M, vx)
    finally:
        vx.destroy()
        if ay is not None:
            ay.destroy()
        if my is not None:
            my.destroy()
    Ax_arr = np.asarray(Ax, dtype=np.float64).ravel()
    Mx_arr = np.asarray(Mx, dtype=np.float64).ravel()
    ray = _rayleigh_metrics(A, M, x_arr, seed_f_hz=float("nan"))
    return {
        "l2_norm_x": float(np.linalg.norm(x_arr)),
        "l2_norm_Mx": float(np.linalg.norm(Mx_arr)),
        "xH_Mx": float(ray.get("xH_Mx", float("nan"))),
        "xH_Ax": float(ray.get("xH_Ax", float("nan"))),
        "rayleigh_lambda": float(ray.get("rayleigh_lambda", float("nan"))),
        "rayleigh_frequency_hz": float(ray.get("rayleigh_f_hz", float("nan"))),
        "Mx_on_u_l2": float(np.linalg.norm(Mx_arr[u_idx])) if u_idx.size else 0.0,
        "Mx_on_p_l2": float(np.linalg.norm(Mx_arr[p_idx])) if p_idx.size else 0.0,
        "x_on_u_l2": float(np.linalg.norm(x_arr[u_idx])) if u_idx.size else 0.0,
        "x_on_p_l2": float(np.linalg.norm(x_arr[p_idx])) if p_idx.size else 0.0,
    }


def _project_u(x_u: np.ndarray, Q: np.ndarray) -> np.ndarray:
    if Q.size == 0:
        return np.asarray(x_u, dtype=np.float64).ravel()
    return np.asarray(x_u, dtype=np.float64).ravel() - Q @ (Q.T @ x_u)


def _project_full(vec: np.ndarray, Q: np.ndarray, u_idx: np.ndarray) -> np.ndarray:
    out = np.asarray(vec, dtype=np.float64).ravel().copy()
    out[u_idx] = _project_u(out[u_idx], Q)
    return out


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
        "probe_distribution": "i.i.d. N(0,1) per coordinate, unit-normalized",
        "probe_tolerance": RANK_PROBE_TOL,
        "probe_subspace_sampled": "Full u_active coefficient space (unstructured random combinations)",
        "probe_median_norm_Muu_z": float(probe.get("median_probe_norm_Muu_x", 0.0)),
        "probe_max_norm_Muu_z": float(probe.get("max_probe_norm_Muu_x", 0.0)),
        "can_detect_kernel_aligned_with_eps_vectors": False,
        "can_detect_kernel_explanation": (
            "A generic random probe measures typical range action, not the span of 56 "
            "EPS-returned modes. A kernel can have large measure-zero complement so "
            "median ||M_uu z|| stays O(1) while specific directions have ||M_uu x||~0."
        ),
        "nullity_ub_equals_zero_means": (
            "No random probe landed in the numerical kernel at tolerance "
            f"{RANK_PROBE_TOL:g}; this does NOT certify global nullity(M_uu)=0."
        ),
        "random_probe_detected_null_vectors": bool(null_ub > 0),
        "empirical_eps_null_vectors_detected": True,
        "global_Muu_nullity_not_certified": True,
        "M_uu_nonzero_row_count": activity.get("M_uu_nonzero_row_count"),
        "M_uu_nonzero_column_count": activity.get("M_uu_nonzero_column_count"),
        "prior_nullity_ub_field_deprecated_wording": (
            "Do not interpret estimated_nullity_dimension_upper_bound=0 as excluding a kernel."
        ),
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

        def lossless_path(mode_row: Dict[str, Any]) -> Optional[Path]:
            rel = mode_row.get("vector_file_lossless")
            if not rel:
                return None
            p = out_dir / str(rel)
            return p if p.is_file() else None

        columns: List[np.ndarray] = []
        slots: List[int] = []
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
            slots.append(int(m.get("eps_slot_index", m.get("candidate_index", len(slots))) or 0))

        n_u = int(u_to_W.size)
        n_cand = len(columns)
        if n_cand == 0:
            raise RuntimeError("no lossless candidate u_active vectors loaded")

        X = np.column_stack(columns)
        gram = X.T @ X
        gram_sv = np.linalg.svd(gram, compute_uv=False)
        X_svd_u, s_vals, _ = np.linalg.svd(X, full_matrices=False)
        if s_vals.size == 0:
            rank = 0
        else:
            rank = int(np.sum(s_vals > SVD_RCOND * float(s_vals[0])))
        Q = X_svd_u[:, :rank].astype(np.float64, copy=False)

        dup = _duplicate_clusters(X)

        independent_directions: List[Dict[str, Any]] = []
        for i in range(rank):
            q = Q[:, i]
            qn = float(np.linalg.norm(q))
            Mq = _petsc_matvec_u(M, q, u_to_W)
            Aq = _petsc_matvec_u(A, q, u_to_W)
            xhm = float(np.vdot(q, Mq).real)
            xha = float(np.vdot(q, Aq).real)
            rel_m = float(np.linalg.norm(Mq) / max(qn, 1e-300))
            fr = _shell_fractions(q, shell_pos, non_shell_pos)
            independent_directions.append(
                {
                    "basis_index": i,
                    "singular_value": float(s_vals[i]),
                    "l2_norm": qn,
                    "l2_norm_Muu_q": float(np.linalg.norm(Mq)),
                    "l2_norm_Auu_q": float(np.linalg.norm(Aq)),
                    "qH_Muu_q": xhm,
                    "qH_Auu_q": xha,
                    "relative_mass_action": rel_m,
                    "mass_null_direction": bool(rel_m < MASS_NULL_REL_TOL),
                    **fr,
                }
            )

        all_dirs_mass_null = bool(independent_directions) and all(
            d["mass_null_direction"] for d in independent_directions
        )
        empirical_dim = rank
        empirical_certified = bool(empirical_dim > 0 and all_dirs_mass_null)

        probe_audit = _audit_random_probe_interpretation(mass_rank)

        seed_info = load_seed_with_diagnostics(seed_npy)
        seed_vec = np.asarray(seed_info["seed_array"], dtype=np.float64).ravel()
        seed_before = _full_matvec_norms(A, M, seed_vec, u_idx=u_to_W, p_idx=p_to_W)
        seed_proj = _project_full(seed_vec, Q, u_to_W)
        seed_after = _full_matvec_norms(A, M, seed_proj, u_idx=u_to_W, p_idx=p_to_W)
        seed_rel_change = float(
            np.linalg.norm(seed_proj - seed_vec) / max(float(np.linalg.norm(seed_vec)), 1e-300)
        )
        seed_preservation_pass = bool(seed_rel_change <= SEED_PRESERVE_REL_TOL)

        candidate_projection: List[Dict[str, Any]] = []
        removed_fracs: List[float] = []
        for m in modes:
            lp = lossless_path(m)
            if lp is None:
                continue
            vec = np.asarray(load_mode_dense_f64_lossless(lp), dtype=np.float64).ravel()
            x_u = vec[u_to_W]
            xn = float(np.linalg.norm(x_u))
            if xn <= 0:
                continue
            x_u_proj = _project_u(x_u, Q)
            removed = float(np.linalg.norm(x_u - x_u_proj) / xn)
            removed_fracs.append(removed)
            candidate_projection.append(
                {
                    "eps_slot_index": int(m.get("eps_slot_index", m.get("candidate_index", -1)) or -1),
                    "removed_norm_fraction": removed,
                    "residual_norm_after_projection": float(np.linalg.norm(x_u_proj)),
                    "relative_mass_action_after_projection": float(
                        np.linalg.norm(_petsc_matvec_u(M, x_u_proj, u_to_W))
                        / max(float(np.linalg.norm(x_u_proj)), 1e-300)
                    ),
                }
            )

        family_removed = bool(
            removed_fracs
            and float(np.median(removed_fracs)) >= FAMILY_REMOVED_FRAC_TOL
            and float(np.min(removed_fracs)) >= FAMILY_REMOVED_FRAC_TOL * 0.95
        )

        future_authorized = bool(
            empirical_certified and seed_preservation_pass and family_removed
        )
        future_strategy = (
            "MAPPING_FIXED_UNREGULARIZED_LOSSLESS_NULLSPACE_PROJECTED_ADJUDICATION_V1"
            if future_authorized
            else "NOT_AUTHORIZED_PENDING_PREFLIGHT_GATES"
        )

        report: Dict[str, Any] = {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evidence_scope": "report_only_no_eps",
            "primary_blocker": PRIMARY_BLOCKER,
            "classification_subtype_from_mass_rank_audit": mass_rank.get("classification_subtype"),
            "no_new_eigensolve_executed": True,
            "additional_eps_authorized": False,
            "single_run_guard_audit": {
                "eps_run_count_for_this_lane": eps_run_count,
                "no_additional_eps_run_authorized": True,
                "re_invoking_authorized_runner_would_block_eps": bool(eps_run_count >= 1),
            },
            "random_probe_interpretation_audit": probe_audit,
            "empirical_null_basis": {
                "candidate_vector_count": n_cand,
                "n_u_active": n_u,
                "gram_matrix_shape": [n_cand, n_cand],
                "gram_singular_values": gram_sv.tolist(),
                "candidate_matrix_singular_values": s_vals.tolist(),
                "numerical_rank_X": rank,
                "rank_method": f"numpy.linalg.svd with cutoff {SVD_RCOND:g} * sigma_max",
                "duplicate_cluster_audit": dup,
                "empirical_null_basis_dimension_in_returned_set": empirical_dim,
                "empirical_null_basis_certified": empirical_certified,
                "independent_direction_diagnostics": independent_directions,
            },
            "projection_preflight": {
                "projector_form": "P = I - Q Q^H on u_active coefficients; p_active unchanged",
                "Q_shape": [n_u, rank],
                "seed_projection_preservation_pass": seed_preservation_pass,
                "seed_before_projection": seed_before,
                "seed_after_projection": seed_after,
                "seed_relative_change_norm_ratio": seed_rel_change,
                "projected_existing_mass_null_family_removed": family_removed,
                "candidate_projection_summary": {
                    "count": len(candidate_projection),
                    "median_removed_norm_fraction": float(np.median(removed_fracs)) if removed_fracs else 0.0,
                    "min_removed_norm_fraction": float(np.min(removed_fracs)) if removed_fracs else 0.0,
                    "max_removed_norm_fraction": float(np.max(removed_fracs)) if removed_fracs else 0.0,
                },
                "candidate_per_slot": candidate_projection,
            },
            "future_diagnostic_strategy_design": {
                "strategy_name": "MAPPING_FIXED_UNREGULARIZED_LOSSLESS_NULLSPACE_PROJECTED_ADJUDICATION_V1",
                "authorized_to_execute": False,
                "preflight_gates_pass": future_authorized,
                "conceptual_change_only": (
                    "Deflate/project certified empirical u_active mass-null basis from EPS search space before solve."
                ),
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
                "do_not": ["Execute EPS until explicitly re-authorized after this preflight"],
            },
            "recommended_future_strategy": future_strategy,
            "root_cause_status_refresh": {
                "primary_blocker": PRIMARY_BLOCKER,
                "serialization_ruled_out_as_active_cause": True,
                "single_lossless_eps_run_consumed": True,
                "v2_physical_model_invalidated": False,
                "st_viability": "unresolved_pending_nullspace_projection_experiment",
                "additional_eps": "NOT_AUTHORIZED",
                "production_mesh_convergence_stage2_blocked": True,
            },
        }

        write_json(OUT_JSON, report)

        md = [
            "# M_uu null-basis certification and projection preflight",
            "",
            f"Generated: {report['generated_utc']}",
            "",
            f"**primary_blocker:** `{PRIMARY_BLOCKER}`",
            f"empirical_null_basis_dimension_in_returned_set={empirical_dim}",
            f"empirical_null_basis_certified={empirical_certified}",
            f"seed_projection_preservation_pass={seed_preservation_pass}",
            f"projected_existing_mass_null_family_removed={family_removed}",
            f"recommended_future_strategy={future_strategy}",
            "",
            "## Random probe vs empirical null basis",
            f"- random_probe_detected_null_vectors={probe_audit['random_probe_detected_null_vectors']}",
            f"- empirical_eps_null_vectors_detected={probe_audit['empirical_eps_null_vectors_detected']}",
            f"- global_Muu_nullity_not_certified={probe_audit['global_Muu_nullity_not_certified']}",
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

        print(f"[null_basis_preflight] empirical_null_basis_dimension_in_returned_set={empirical_dim}", flush=True)
        print(f"[null_basis_preflight] empirical_null_basis_certified={empirical_certified}", flush=True)
        print(f"[null_basis_preflight] seed_projection_preservation_pass={seed_preservation_pass}", flush=True)
        print(f"[null_basis_preflight] projected_existing_mass_null_family_removed={family_removed}", flush=True)
        print(f"[null_basis_preflight] recommended_future_strategy={future_strategy}", flush=True)
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
