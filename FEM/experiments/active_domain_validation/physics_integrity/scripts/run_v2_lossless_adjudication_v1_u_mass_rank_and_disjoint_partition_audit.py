#!/usr/bin/env python3
"""
Report-only: disjoint u_active partition + structural M_uu rank/null-space audit for lossless adjudication v1.

Reads existing isolated lossless artifacts and reassembled replay operators only.
Does not call eps.solve() or modify solver policy.
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

OUT_JSON = CONV_DIAG / "v2_lossless_adjudication_v1_u_mass_rank_and_disjoint_partition_audit.json"
OUT_MD = CONV_DIAG / "v2_lossless_adjudication_v1_u_mass_rank_and_disjoint_partition_audit.md"
PRIOR_ATTRIBUTION_JSON = CONV_DIAG / "v2_lossless_adjudication_v1_u_active_nullspace_attribution.json"

ROW_COL_TOL = 1e-15
M_NULL_VEC_TOL = 1e-30
RANK_PROBE_DIM = 48
RANK_PROBE_TOL = 1e-10

CLASSIFICATION_PARENT = "UNRESOLVED_U_ACTIVE_NULLSPACE"


def _np_int32_1d(raw: Any) -> np.ndarray:
    if raw is None:
        return np.asarray([], dtype=np.int32)
    if hasattr(raw, "array") and not isinstance(raw, (list, tuple, np.ndarray)):
        try:
            raw = raw.array
        except Exception:
            pass
    return np.asarray(raw, dtype=np.int32).ravel()


def _coalesce_list(*candidates: Any) -> list:
    for c in candidates:
        if isinstance(c, list):
            return c
    return []


def _atomic_load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _mat_row_cols_vals(M: Any, row: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        cols, vals = M.getRow(int(row))
    except TypeError:
        got = M.getRow(int(row))
        cols, vals = got[0], got[1]
    return np.asarray(cols, dtype=np.int32).ravel(), np.asarray(vals, dtype=np.float64).ravel()


def _build_disjoint_u_partition(
    tag_subsets: Dict[str, np.ndarray],
    u_to_W: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Mutually exclusive elimination partition of reduced u_active indices (global W local indices)."""
    u_all = np.unique(np.asarray(u_to_W, dtype=np.int32).ravel())
    tag5 = np.unique(np.asarray(tag_subsets["tag_5_pinned_fix_displacement"], dtype=np.int32))
    shell_union = np.unique(
        np.concatenate(
            [
                tag_subsets["tag_1_top_shell_displacement"],
                tag_subsets["tag_3_back_shell_displacement"],
                tag_subsets["tag_4_ribs_side_displacement"],
            ]
        ).astype(np.int32, copy=False)
    )
    shell_phys_excl_tag5 = (
        np.setdiff1d(shell_union, tag5, assume_unique=True).astype(np.int32)
        if shell_union.size
        else np.array([], dtype=np.int32)
    )

    u_tag5_only = tag5.astype(np.int32, copy=False)
    u_shell_phys = shell_phys_excl_tag5.astype(np.int32, copy=False)
    assigned = np.unique(np.concatenate([u_tag5_only, u_shell_phys]).astype(np.int32, copy=False))
    u_non_shell = np.setdiff1d(u_all, assigned, assume_unique=True).astype(np.int32)

    overlap_tag5_shell_raw = np.intersect1d(tag5, shell_union).astype(np.int32, copy=False)
    overlap_assigned = np.intersect1d(u_tag5_only, u_shell_phys).astype(np.int32, copy=False)

    partition = {
        "u_tag5_fix_only": u_tag5_only,
        "u_shell_physical_union_excluding_tag5": u_shell_phys,
        "u_non_shell_complement": u_non_shell,
        "u_unclassified_or_overlap_error": overlap_assigned.astype(np.int32, copy=False),
    }

    elimination_keys = (
        "u_tag5_fix_only",
        "u_shell_physical_union_excluding_tag5",
        "u_non_shell_complement",
        "u_unclassified_or_overlap_error",
    )
    union_parts = np.unique(np.concatenate([partition[k] for k in elimination_keys]).astype(np.int32, copy=False))
    pairwise = {
        "u_tag5_fix_only_vs_u_shell_physical_union_excluding_tag5": int(
            np.intersect1d(partition["u_tag5_fix_only"], partition["u_shell_physical_union_excluding_tag5"]).size
        ),
        "u_tag5_fix_only_vs_u_non_shell_complement": int(
            np.intersect1d(partition["u_tag5_fix_only"], partition["u_non_shell_complement"]).size
        ),
        "u_shell_physical_union_excluding_tag5_vs_u_non_shell_complement": int(
            np.intersect1d(
                partition["u_shell_physical_union_excluding_tag5"], partition["u_non_shell_complement"]
            ).size
        ),
    }
    meta = {
        "n_u_active": int(u_all.size),
        "partition_counts": {k: int(partition[k].size) for k in elimination_keys},
        "union_count": int(union_parts.size),
        "pairwise_overlap_counts": pairwise,
        "raw_tag_overlaps": {
            "tag5_vs_shell_union_raw": int(overlap_tag5_shell_raw.size),
            "tag1_vs_tag3": int(
                np.intersect1d(
                    tag_subsets["tag_1_top_shell_displacement"],
                    tag_subsets["tag_3_back_shell_displacement"],
                ).size
            ),
            "tag1_vs_tag4": int(
                np.intersect1d(
                    tag_subsets["tag_1_top_shell_displacement"],
                    tag_subsets["tag_4_ribs_side_displacement"],
                ).size
            ),
            "tag3_vs_tag4": int(
                np.intersect1d(
                    tag_subsets["tag_3_back_shell_displacement"],
                    tag_subsets["tag_4_ribs_side_displacement"],
                ).size
            ),
        },
        "raw_tag_overlaps_note": (
            "Facet tag unions may overlap in collapsed u space; the elimination partition is "
            "mutually exclusive by construction (tag5 first, then shell\\tag5, then complement)."
        ),
        "partition_pass": bool(
            union_parts.size == u_all.size
            and all(v == 0 for v in pairwise.values())
            and partition["u_unclassified_or_overlap_error"].size == 0
        ),
    }
    return partition, meta


def _petsc_matvec_u(M: Any, x_u: np.ndarray, u_idx: np.ndarray) -> np.ndarray:
    from physical_fsi_seed_residual_audit import _petsc_matvec, _petsc_vec_from_array

    n = int(M.getSize()[1])
    x_full = np.zeros(n, dtype=np.float64)
    x_full[np.asarray(u_idx, dtype=np.int32)] = np.asarray(x_u, dtype=np.float64).ravel()
    vx = _petsc_vec_from_array(M, x_full)
    try:
        y, vy = _petsc_matvec(M, vx)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
    finally:
        vx.destroy()
        if vy is not None:
            vy.destroy()
    return y_arr[np.asarray(u_idx, dtype=np.int32)]


def _partition_row_inactivity_frac(M: Any, part_idx: np.ndarray, u_set: set, tol: float = ROW_COL_TOL) -> float:
    part_idx = np.asarray(part_idx, dtype=np.int32).ravel()
    if part_idx.size == 0:
        return 1.0
    inactive = 0
    for row in part_idx:
        cols, vals = _mat_row_cols_vals(M, int(row))
        if cols.size == 0:
            inactive += 1
            continue
        active = any(abs(float(v)) > tol and int(c) in u_set for c, v in zip(cols, vals))
        if not active:
            inactive += 1
    return float(inactive / part_idx.size)


def _muu_row_col_activity(M: Any, u_idx: np.ndarray, tol: float = ROW_COL_TOL) -> Dict[str, Any]:
    u_idx = np.asarray(u_idx, dtype=np.int32).ravel()
    n_u = int(u_idx.size)
    u_set = {int(i) for i in u_idx.tolist()}

    active_rows: set = set()
    active_cols: set = set()
    for row in u_idx:
        cols, vals = _mat_row_cols_vals(M, int(row))
        for c, v in zip(cols, vals):
            if abs(float(v)) > tol and int(c) in u_set:
                active_rows.add(int(row))
                active_cols.add(int(c))

    return {
        "n_u_active": n_u,
        "M_uu_nonzero_row_count": len(active_rows),
        "M_uu_nonzero_column_count": len(active_cols),
        "rows_with_nonzero_on_u_columns_frac": float(len(active_rows) / max(n_u, 1)),
        "columns_with_nonzero_on_u_rows_frac": float(len(active_cols) / max(n_u, 1)),
        "mass_active_geometric_partition_vs_algebraic_kernel": (
            "Column participates in M (nonzero row/column in M_uu block) does NOT imply "
            "a vector lies outside null(M_uu); kernel vectors can cancel on participating coordinates."
        ),
        "rank_nullity_method": (
            "Randomized range probe: sample unit random z in R^{n_u_active}, "
            f"record ||M_uu z||; count probes with norm < {RANK_PROBE_TOL:g} as nullity upper bound."
        ),
    }


def _estimate_muu_nullity_probe(M: Any, u_idx: np.ndarray, dim: int = RANK_PROBE_DIM) -> Dict[str, Any]:
    rng = np.random.default_rng(0)
    n_u = int(u_idx.size)
    if n_u == 0:
        return {"probe_dim": 0, "median_probe_norm_Muu_x": 0.0, "estimated_nullity_dimension_upper_bound": 0}
    n_probe = min(dim, max(8, n_u // 5000))
    norms: List[float] = []
    for _ in range(n_probe):
        z = rng.standard_normal(n_u)
        zn = float(np.linalg.norm(z))
        if zn <= 0:
            continue
        z /= zn
        y = _petsc_matvec_u(M, z, u_idx)
        norms.append(float(np.linalg.norm(y)))
    med = float(np.median(norms)) if norms else 0.0
    null_ub = int(sum(1 for n in norms if n < RANK_PROBE_TOL))
    return {
        "probe_dim": int(len(norms)),
        "median_probe_norm_Muu_x": med,
        "max_probe_norm_Muu_x": float(max(norms)) if norms else 0.0,
        "estimated_nullity_dimension_upper_bound": null_ub,
        "numerical_rank_estimate_lower_bound": int(max(len(norms) - null_ub, 0)),
    }


def _fraction_stats_on_partition(vec: np.ndarray, partition: Dict[str, np.ndarray]) -> Dict[str, Any]:
    l2 = float(np.linalg.norm(vec))
    out: Dict[str, Any] = {}
    for name, idx in partition.items():
        idx = np.asarray(idx, dtype=np.int32).ravel()
        if idx.size == 0:
            out[name] = {"l2_norm": 0.0, "l2_fraction": 0.0, "max_abs": 0.0}
            continue
        blk = vec[idx]
        out[name] = {
            "l2_norm": float(np.linalg.norm(blk)),
            "l2_fraction": float(np.linalg.norm(blk) / max(l2, 1e-300)),
            "max_abs": float(np.max(np.abs(blk))) if blk.size else 0.0,
        }
    return out


def _agg_min_median_max(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {"min": float(np.min(arr)), "median": float(np.median(arr)), "max": float(np.max(arr))}


def _classify_subtype(
    *,
    partition_meta: Dict[str, Any],
    m_uu_activity: Dict[str, Any],
    candidate_agg: Dict[str, Any],
    non_shell_inactive_row_frac: float,
    shell_phys_row_active_frac: float,
) -> Tuple[str, str, str]:
    med_non_shell = float(candidate_agg.get("median_l2_fraction_non_shell", 0.0))
    med_tag5 = float(candidate_agg.get("median_l2_fraction_tag5", 0.0))
    med_shell_excl = float(candidate_agg.get("median_l2_fraction_shell_excl_tag5", 0.0))
    med_rel_mass = float(candidate_agg.get("median_relative_mass_action_Muu", 0.0))
    row_active_frac = float(m_uu_activity.get("rows_with_nonzero_on_u_columns_frac", 0.0))

    if not partition_meta.get("partition_pass"):
        return (
            CLASSIFICATION_PARENT,
            "U_NULLSPACE_ATTRIBUTION_UNRESOLVED",
            "partition_disjoint_check_failed",
        )

    if med_tag5 >= 0.5:
        return (
            CLASSIFICATION_PARENT,
            "U_NULLSPACE_ATTRIBUTION_UNRESOLVED",
            "tag5_fraction_high_on_disjoint_partition",
        )

    if (
        med_non_shell >= 0.85
        and non_shell_inactive_row_frac >= 0.85
        and med_rel_mass < 1e-8
    ):
        return (
            CLASSIFICATION_PARENT,
            "U_NULLSPACE_NON_SHELL_INACTIVE_DOF_DOMINATED",
            (
                "Candidates concentrate on disjoint non-shell u DOFs whose rows are largely inactive "
                "in M_uu, with negligible ||M_uu x_u||."
            ),
        )

    if (
        med_rel_mass < 1e-6
        and shell_phys_row_active_frac >= 0.25
        and non_shell_inactive_row_frac < 0.85
    ):
        return (
            CLASSIFICATION_PARENT,
            "U_NULLSPACE_SHELL_MASS_MATRIX_KERNEL",
            (
                "Candidates exhibit negligible restricted mass action while shell-participating "
                "coordinates remain active in M_uu (kernel/cancellation, not inactive DOFs alone)."
            ),
        )

    if med_non_shell >= 0.5 and med_shell_excl >= 0.2 and med_rel_mass < 1e-6:
        return (
            CLASSIFICATION_PARENT,
            "U_NULLSPACE_MIXED_INACTIVE_AND_KERNEL",
            "Both non-shell concentration and shell-participating kernel effects appear material.",
        )

    if med_rel_mass < 1e-6 and row_active_frac > 0.5:
        return (
            CLASSIFICATION_PARENT,
            "U_NULLSPACE_SHELL_MASS_MATRIX_KERNEL",
            (
                "Global M_uu row participation is high but all candidates have near-zero ||M_uu x_u|| "
                "(consistent with rank-deficient mass block / kernel, not serialization)."
            ),
        )

    return (
        CLASSIFICATION_PARENT,
        "U_NULLSPACE_ATTRIBUTION_UNRESOLVED",
        "Disjoint partition and M_uu activity do not support a single dominant mechanism.",
    )


def _remediation_for_subtype(subtype: str) -> Dict[str, Any]:
    if subtype == "U_NULLSPACE_NON_SHELL_INACTIVE_DOF_DOMINATED":
        return {
            "recommended_option": (
                "Restrict structural displacement coordinates to physical shell-support DOFs before EPS"
            ),
            "rationale": (
                "Mass-null vectors concentrate on displacement DOFs outside the shell facet union with "
                "negligible M_uu row coupling on that subset."
            ),
            "do_not": ["Remove shell DOFs by geometric tag alone without rank verification"],
        }
    if subtype == "U_NULLSPACE_SHELL_MASS_MATRIX_KERNEL":
        return {
            "recommended_option": (
                "Nullspace projection/deflation or physical mass-bearing range formulation on M_uu"
            ),
            "rationale": (
                "Vectors occupy shell-related coordinates but lie in the numerical kernel of the restricted mass block."
            ),
            "do_not": ["Delete shell DOFs by tag alone"],
        }
    if subtype == "U_NULLSPACE_MIXED_INACTIVE_AND_KERNEL":
        return {
            "recommended_option": (
                "Rank-revealing / nullspace-basis diagnostics before any DOF elimination"
            ),
            "rationale": "Inactive-DOF and kernel effects both contribute; need separation before projection.",
            "do_not": ["Implement projection yet", "Authorize another EPS solve"],
        }
    return {
        "recommended_option": "Further diagnostics required before DOF elimination or EPS policy change",
        "rationale": subtype,
        "do_not": ["Authorize another EPS solve", "Implement projection yet"],
    }


def main() -> int:
    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[mass_rank_audit] Requires mpiexec -n 1", file=sys.stderr)
        return 2

    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
    case_dir = solve_case_dir("L_mid", CASE_ID)
    out_dir = case_dir / OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1
    mesh_file = mesh_path("L_mid", CASE_ID)
    seed_npy = case_dir / "diagnostics" / "acoustic_coupled_seed.npy"

    prior = _atomic_load_json(PRIOR_ATTRIBUTION_JSON)
    prior_ev = (prior.get("refined_classification") or {}).get("evidence") or prior.get("evidence") or {}
    eps_auth = _atomic_load_json(CONV_DIAG / "v2_lossless_adjudication_v1_eps_authorization_record.json")
    eps_run_count = int(eps_auth.get("eps_run_count_for_this_lane", 0) or 0)

    sample = sample_spec_from_case(case)
    A, M, cfg = _assemble_reduced_coupled_replay(mesh_file, sample, coupling_enabled=True)
    maps = _extract_layout_maps(cfg, A)
    u_to_W = _np_int32_1d(maps.get("u_to_W"))
    p_to_W = _np_int32_1d(maps.get("p_to_W"))
    operator_size = int(A.getSize()[0])
    u_set = {int(i) for i in u_to_W.tolist()}

    try:
        from run_v2_lossless_adjudication_v1_u_active_nullspace_attribution import (
            _build_tag_subsets_in_reduced_u,
        )

        tag_map = _build_tag_subsets_in_reduced_u(
            mesh_file=mesh_file,
            sample=sample,
            u_to_W=u_to_W,
            p_to_W=p_to_W,
            operator_size=operator_size,
        )
        partition, partition_meta = _build_disjoint_u_partition(tag_map["subsets"], u_to_W)

        modes = _coalesce_list(_atomic_load_json(out_dir / "diagnostics/mode_energy_summary.json").get("modes"))
        from fem_mode_array_utils import load_mode_dense_f64_lossless

        def lossless_path(mode_row: Dict[str, Any]) -> Optional[Path]:
            rel = mode_row.get("vector_file_lossless")
            if not rel:
                return None
            p = out_dir / str(rel)
            return p if p.is_file() else None

        seed_info = load_seed_with_diagnostics(seed_npy)
        if seed_info.get("seed_load_status") != "ok" or not isinstance(seed_info.get("seed_array"), np.ndarray):
            raise RuntimeError("seed_array missing or invalid")
        seed_vec = np.asarray(seed_info["seed_array"], dtype=np.float64).ravel()

        seed_x_u = seed_vec[u_to_W]
        seed_Mxu = _petsc_matvec_u(M, seed_x_u, u_to_W)
        seed_rel_mass = float(np.linalg.norm(seed_Mxu) / max(float(np.linalg.norm(seed_x_u)), 1e-300))

        m_uu_activity = _muu_row_col_activity(M, u_to_W)
        nullity_probe = _estimate_muu_nullity_probe(M, u_to_W)

        non_shell_inactive_row_frac = _partition_row_inactivity_frac(
            M, partition["u_non_shell_complement"], u_set
        )
        shell_phys_row_active_frac = 1.0 - _partition_row_inactivity_frac(
            M, partition["u_shell_physical_union_excluding_tag5"], u_set
        )

        per_candidate: List[Dict[str, Any]] = []
        frac_by_part: Dict[str, List[float]] = {k: [] for k in partition}
        rel_mass_list: List[float] = []

        for m in modes:
            slot = int(m.get("eps_slot_index", m.get("candidate_index", m.get("mode_index", 0))) or 0)
            lp = lossless_path(m)
            if lp is None:
                continue
            vec = np.asarray(load_mode_dense_f64_lossless(lp), dtype=np.float64).ravel()
            x_u = vec[u_to_W]
            xn = float(np.linalg.norm(x_u))
            if xn <= 0:
                continue
            Mxu = _petsc_matvec_u(M, x_u, u_to_W)
            rel_mass = float(np.linalg.norm(Mxu) / xn)
            rel_mass_list.append(rel_mass)
            part_fracs = _fraction_stats_on_partition(vec, partition)
            for k, st in part_fracs.items():
                frac_by_part.setdefault(k, []).append(float(st["l2_fraction"]))
            per_candidate.append(
                {
                    "eps_slot_index": slot,
                    "partition_l2_fractions": part_fracs,
                    "relative_mass_action_Muu": rel_mass,
                    "mass_null_by_absolute": bool(rel_mass < 1e-12 and float(np.linalg.norm(Mxu)) < M_NULL_VEC_TOL),
                }
            )

        partition_fraction_aggregate = {
            k: _agg_min_median_max(v) for k, v in frac_by_part.items() if k in partition
        }
        candidate_agg = {
            "count": len(per_candidate),
            "median_l2_fraction_non_shell": float(
                partition_fraction_aggregate.get("u_non_shell_complement", {}).get("median", 0.0)
            ),
            "median_l2_fraction_tag5": float(
                partition_fraction_aggregate.get("u_tag5_fix_only", {}).get("median", 0.0)
            ),
            "median_l2_fraction_shell_excl_tag5": float(
                partition_fraction_aggregate.get("u_shell_physical_union_excluding_tag5", {}).get("median", 0.0)
            ),
            "median_relative_mass_action_Muu": float(np.median(rel_mass_list)) if rel_mass_list else 0.0,
            "partition_fraction_min_median_max": partition_fraction_aggregate,
            "median_overlap_on_M_active_u_cols_prior_attribution": float(
                prior_ev.get("median_overlap_on_M_active_u_cols", 0.0)
            ),
            "median_overlap_on_M_null_u_cols_prior_attribution": float(
                prior_ev.get("median_overlap_on_M_null_u_cols", 0.0)
            ),
        }

        classification, subtype, reason = _classify_subtype(
            partition_meta=partition_meta,
            m_uu_activity=m_uu_activity,
            candidate_agg=candidate_agg,
            non_shell_inactive_row_frac=non_shell_inactive_row_frac,
            shell_phys_row_active_frac=shell_phys_row_active_frac,
        )
        remediation = _remediation_for_subtype(subtype)

        dominant = "u_non_shell_complement"
        if subtype == "U_NULLSPACE_SHELL_MASS_MATRIX_KERNEL":
            dominant = "u_shell_physical_union_excluding_tag5"
        elif subtype == "U_NULLSPACE_NON_SHELL_INACTIVE_DOF_DOMINATED":
            dominant = "u_non_shell_complement"

        rank_summary = (
            f"rows={m_uu_activity['M_uu_nonzero_row_count']}/{m_uu_activity['n_u_active']} "
            f"cols={m_uu_activity['M_uu_nonzero_column_count']}/{m_uu_activity['n_u_active']} "
            f"probe_median={nullity_probe['median_probe_norm_Muu_x']:.3e} "
            f"nullity_ub={nullity_probe['estimated_nullity_dimension_upper_bound']}"
        )

        report: Dict[str, Any] = {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evidence_scope": "report_only_no_eps",
            "classification": classification,
            "classification_subtype": subtype,
            "classification_reason": reason,
            "dominant_support_category": dominant,
            "no_new_eigensolve_executed": True,
            "additional_eps_authorized": False,
            "M_uu_nullity_or_rank_summary": rank_summary,
            "single_run_guard_audit": {
                "eps_run_count_for_this_lane": eps_run_count,
                "no_additional_eps_run_authorized": True,
                "re_invoking_authorized_runner_would_block_eps": bool(eps_run_count >= 1),
            },
            "disjoint_u_active_partition": {
                "partition": {k: v.tolist() for k, v in partition.items()},
                "partition_meta": partition_meta,
            },
            "structural_M_uu_activity": {
                **m_uu_activity,
                "non_shell_partition_row_inactivity_frac": non_shell_inactive_row_frac,
                "shell_phys_partition_row_activity_frac": shell_phys_row_active_frac,
            },
            "M_uu_nullity_probe": nullity_probe,
            "seed_control": {
                "seed_relative_mass_action_Muu_on_u_block": seed_rel_mass,
                "note": "Full coupled seed is p_active-dominated; do not treat as structural M_uu test vector.",
            },
            "candidate_aggregate_partition_fractions": candidate_agg,
            "candidate_per_slot": per_candidate,
            "candidate_seed_M_action_ratio_summary": {
                "median_relative_mass_action_Muu": candidate_agg["median_relative_mass_action_Muu"],
                "max_relative_mass_action_Muu": float(max(rel_mass_list)) if rel_mass_list else 0.0,
                "count_mass_null_absolute": int(sum(1 for c in per_candidate if c.get("mass_null_by_absolute"))),
            },
            "remediation_design": remediation,
            "root_cause_status_refresh": {
                "single_lossless_eps_run_consumed": True,
                "serialization_ruled_out_as_active_cause": True,
                "current_blocker": "u_active null(M) attribution unresolved",
                "additional_eps": "NOT_AUTHORIZED",
                "v2_physical_model_invalidated": False,
            },
        }

        write_json(OUT_JSON, report)

        md_lines = [
            "# Lossless adjudication v1: mass-rank and disjoint-partition audit",
            "",
            f"Generated: {report['generated_utc']}",
            "",
            f"**classification:** `{classification}`",
            f"**classification_subtype:** `{subtype}`",
            f"**classification_reason:** {reason}",
            f"**dominant_support_category:** `{dominant}`",
            "",
            f"partition_pass={partition_meta.get('partition_pass')}",
            f"M_uu_nullity_or_rank_summary={rank_summary}",
            f"median_candidate_relative_mass_action_Muu={candidate_agg['median_relative_mass_action_Muu']}",
            f"recommended_option={remediation.get('recommended_option')}",
            "",
            "## Status",
            "- single lossless EPS run consumed",
            "- serialization ruled out as active cause",
            "- current blocker = u_active null(M) attribution unresolved",
            "- additional EPS = NOT AUTHORIZED",
            "- V2 physical model = not invalidated",
            "",
        ]
        OUT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

        try:
            from write_v2_st_singular_mass_rehabilitation_plan import main as rehab_main
            from run_v2_solver_root_cause_and_forward_risk_audit import main as audit_main

            rehab_main()
            audit_main()
        except Exception as exc:
            print(f"[mass_rank_audit] status_refresh_warning={type(exc).__name__}:{exc}", flush=True)

        print(f"[mass_rank_audit] partition_pass={partition_meta.get('partition_pass')}", flush=True)
        print(f"[mass_rank_audit] M_uu_nullity_or_rank_summary={rank_summary}", flush=True)
        print(f"[mass_rank_audit] refined_subtype={subtype}", flush=True)
        print(f"[mass_rank_audit] recommended_option={remediation.get('recommended_option')}", flush=True)
        print("[mass_rank_audit] no_new_eigensolve_executed=True", flush=True)
        print("[mass_rank_audit] additional_eps=NOT_AUTHORIZED", flush=True)
        return 0
    finally:
        try:
            A.destroy()
            M.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
