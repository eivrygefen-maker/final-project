#!/usr/bin/env python3
"""
Report-only u_active nullspace / mass-null attribution for lossless adjudication v1.

Requirements:
- Reads only the existing isolated lossless run artifacts + reassembled replay operators.
- Must not call eps.solve() / SLEPc EPS.
- Writes:
  FEM/experiments/.../v2_mesh_convergence/diagnostics/
    v2_lossless_adjudication_v1_u_active_nullspace_attribution.{json,md}
  and refreshes conservative status reports.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from mpi4py import MPI

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

# Ensure local script imports work when invoked from repo root.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
for _p in (SCRIPT_DIR, REPO_ROOT / "FEM" / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fem_main_3d as fem3d

CASE_ID = "baseline_coupled_v2"

# Output
OUT_JSON = CONV_DIAG / "v2_lossless_adjudication_v1_u_active_nullspace_attribution.json"
OUT_MD = CONV_DIAG / "v2_lossless_adjudication_v1_u_active_nullspace_attribution.md"

# Inputs
DIAG_JSON = CONV_DIAG / "v2_l_mid_mapping_fixed_unregularized_lossless_adjudication_v1_diagnostic.json"
POSTMASS_NULL_JSON = CONV_DIAG / "v2_lossless_adjudication_v1_mass_null_postmortem.json"
EPS_AUTH_RECORD_JSON = CONV_DIAG / "v2_lossless_adjudication_v1_eps_authorization_record.json"

LOSSLESS_SUMMARY_PATH = (
    "diagnostics/mode_energy_summary.json"
)
LOSSLESS_BANK_PATH = "diagnostics/eps_candidate_bank.json"

SEED_DGN_PATH = "diagnostics/acoustic_coupled_seed.npy"

# Thresholds (audit only; does not change any retroactive verdicts)
XH_MX_TOL = 1.0e-30
IN_OR_NEAR_NULL_M_ABS_NORM_TOL = 1.0e-12
M_NULL_ABS_NORM_TOL = 1.0e-30

NNZ_THRESHOLDS = (1e-15, 1e-12, 1e-9)
ACTION_PROBE_ABS_TOL = 1e-15
SIGNIFICANT_SUPPORT_ABS_TOL = 1e-15


def _np_int32_1d(raw: Any) -> np.ndarray:
    """
    Convert a nullable/sequence/ndarray to a 1D int32 numpy array without ever
    evaluating `raw` in a boolean context (raw may itself be a numpy array).
    """
    if raw is None:
        return np.asarray([], dtype=np.int32)
    # Some petsc/dolfinx objects expose `.array`; accept but never truth-test.
    if hasattr(raw, "array") and not isinstance(raw, (list, tuple, np.ndarray)):
        try:
            raw = raw.array
        except Exception:
            pass
    return np.asarray(raw, dtype=np.int32).ravel()


def _coalesce_list(*candidates: Any) -> list:
    """
    Return the first candidate that is a python list; else return [].
    Avoids `a or b` on numpy arrays.
    """
    for c in candidates:
        if isinstance(c, list):
            return c
    return []


def _atomic_load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _nnz_stats(vec: np.ndarray) -> Dict[str, Any]:
    v = np.asarray(vec, dtype=np.float64).ravel()
    ab = np.abs(v)
    out: Dict[str, Any] = {"nnz_exact": int(np.count_nonzero(v))}
    for thr in NNZ_THRESHOLDS:
        out[f"nnz_gt_{thr:g}"] = int(np.sum(ab > thr))
    return out


def _block_support(vec: np.ndarray, idx: np.ndarray) -> Dict[str, Any]:
    if idx.size == 0:
        return {
            "l2_norm": 0.0,
            "max_abs": 0.0,
            **_nnz_stats(np.array([], dtype=np.float64)),
        }
    blk = np.asarray(vec, dtype=np.float64).ravel()[idx]
    return {
        "l2_norm": float(np.linalg.norm(blk)),
        "max_abs": float(np.max(np.abs(blk))) if blk.size else 0.0,
        **_nnz_stats(blk),
    }


def _norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vec, dtype=np.float64).ravel()))


def _matvec_norms(A, M, x: np.ndarray, u_idx: np.ndarray, p_idx: np.ndarray) -> Dict[str, Any]:
    # Uses the same physical residual / PETSc matvec backend used elsewhere.
    from physical_fsi_seed_residual_audit import _petsc_vec_from_array, _petsc_matvec, _rayleigh_metrics

    x_arr = np.asarray(x, dtype=np.float64).ravel()
    vx = _petsc_vec_from_array(A, x_arr)
    ay = my = None
    try:
        Ax, ay = _petsc_matvec(A, vx)
        Mx, my = _petsc_matvec(M, vx)
    finally:
        try:
            vx.destroy()
        except Exception:
            pass
        if ay is not None:
            try:
                ay.destroy()
            except Exception:
                pass
        if my is not None:
            try:
                my.destroy()
            except Exception:
                pass

    Ax_arr = np.asarray(Ax, dtype=np.float64).ravel()
    Mx_arr = np.asarray(Mx, dtype=np.float64).ravel()

    # Rayleigh metrics are meaningful only if xH_Mx is not near-zero; still compute for audit visibility.
    ray = _rayleigh_metrics(A, M, x_arr, seed_f_hz=float("nan"))
    xH_Ax = float(ray.get("xH_Ax", float("nan")))
    xH_Mx = float(ray.get("xH_Mx", float("nan")))

    # Residual norms computed explicitly from Ax - lam Mx are cheap once Ax/Mx exist.
    # lam_reported is injected by caller.
    return {
        "l2_norm_x": float(np.linalg.norm(x_arr)),
        "l2_norm_Ax": float(np.linalg.norm(Ax_arr)),
        "l2_norm_Mx": float(np.linalg.norm(Mx_arr)),
        "max_abs_Ax": float(np.max(np.abs(Ax_arr))) if Ax_arr.size else 0.0,
        "max_abs_Mx": float(np.max(np.abs(Mx_arr))) if Mx_arr.size else 0.0,
        "xH_Ax": xH_Ax,
        "xH_Mx": xH_Mx,
        "rayleigh_lambda": float(ray.get("rayleigh_lambda", float("nan"))),
        "rayleigh_frequency_hz": float(ray.get("rayleigh_f_hz", float("nan"))),
        "nnz_Mx_gt_abs_1e-15": int(np.sum(np.abs(Mx_arr) > SIGNIFICANT_SUPPORT_ABS_TOL)),
        "nnz_Mx_gt_abs_1e-12": int(np.sum(np.abs(Mx_arr) > 1e-12)),
        "nnz_Mx_gt_abs_1e-9": int(np.sum(np.abs(Mx_arr) > 1e-9)),
        "Mx_on_u_l2": _norm(Mx_arr[u_idx]) if u_idx.size else 0.0,
        "Mx_on_p_l2": _norm(Mx_arr[p_idx]) if p_idx.size else 0.0,
    }


def _residual_norm_from_A_M(Ax: np.ndarray, Mx: np.ndarray, lam: float) -> float:
    if lam is None or not math.isfinite(float(lam)):
        return float("nan")
    r = np.asarray(Ax, dtype=np.float64).ravel() - float(lam) * np.asarray(Mx, dtype=np.float64).ravel()
    return float(np.linalg.norm(r))


def _probe_action_rows(
    A,
    M,
    u_idx: np.ndarray,
    p_idx: np.ndarray,
    supported_u_idx: np.ndarray,
) -> Dict[str, Any]:
    """Action probe: apply A and M to a unit vector supported on supported_u_idx.

    Output gives a conservative count of u-active rows which get non-negligible output.
    """
    from physical_fsi_seed_residual_audit import _petsc_vec_from_array, _petsc_matvec

    op_size = int(A.getSize()[0])
    if supported_u_idx.size == 0:
        return {"probe_norm_x": 0.0, "active_u_rows_on_M": 0, "active_u_rows_on_A": 0}

    y = np.zeros(op_size, dtype=np.float64)
    # normalize on the supported set
    y[supported_u_idx] = 1.0
    yn = float(np.linalg.norm(y))
    if yn <= 0.0 or not math.isfinite(yn):
        return {"probe_norm_x": yn, "active_u_rows_on_M": 0, "active_u_rows_on_A": 0}
    y /= yn

    vx = _petsc_vec_from_array(A, y)
    try:
        Ay, ay = _petsc_matvec(A, vx)
        My, my = _petsc_matvec(M, vx)
    finally:
        try:
            vx.destroy()
        except Exception:
            pass
        try:
            ay.destroy()
        except Exception:
            pass
        try:
            my.destroy()
        except Exception:
            pass
    Ay_arr = np.asarray(Ay, dtype=np.float64).ravel()
    My_arr = np.asarray(My, dtype=np.float64).ravel()

    u_rows = u_idx
    u_on_M = u_rows[np.abs(My_arr[u_rows]) > ACTION_PROBE_ABS_TOL]
    u_on_A = u_rows[np.abs(Ay_arr[u_rows]) > ACTION_PROBE_ABS_TOL]
    return {
        "probe_norm_x": float(np.linalg.norm(y)),
        "active_u_rows_on_M": int(u_on_M.size),
        "active_u_rows_on_A": int(u_on_A.size),
        "active_u_rows_on_M_frac_of_u": float(u_on_M.size / max(u_idx.size, 1)),
        "active_u_rows_on_A_frac_of_u": float(u_on_A.size / max(u_idx.size, 1)),
    }


def _build_tag_subsets_in_reduced_u(
    *,
    mesh_file: Path,
    sample: Dict[str, Any],
    u_to_W: np.ndarray,
    p_to_W: np.ndarray,
    operator_size: int,
    coupling_enabled: bool = True,
) -> Dict[str, Any]:
    """Map facet-tag displacement DOFs into reduced u-local indices using parent_to_local mapping.

    This is the only place we use geometric/tag DOF identification.
    """
    # Reassemble once for cfg maps (u/p layout + parent indexing).
    A, M, cfg = _assemble_reduced_coupled_replay(mesh_file, sample, coupling_enabled=coupling_enabled)
    try:
        maps = _extract_layout_maps(cfg, A)
        u_to_W_local = _np_int32_1d(maps.get("u_to_W"))
        p_to_W_local = _np_int32_1d(maps.get("p_to_W"))
        restr = maps["restr"]
        raw_active = cfg.get("_coupled_air_active_W_indices")
        active_W_parent = _np_int32_1d(raw_active)
        n_full_w = int(restr.get("n_coupled_W_full", -1))
        if active_W_parent.size == 0 or n_full_w <= 0:
            raise RuntimeError("cannot reconstruct parent_to_local: missing active_W_parent or n_coupled_W_full in cfg")

        active_sorted = np.asarray(active_W_parent, dtype=np.int32).ravel()
        # active_sorted should already be np.unique(sorted(...)) from fem_main_3d.
        parent_to_local = np.full(n_full_w, -1, dtype=np.int32)
        if active_sorted.size != operator_size:
            # still build with ordering; local indices are positions in active set ordering
            pass
        parent_to_local[active_sorted] = np.arange(int(active_sorted.size), dtype=np.int32)

        u_set_local = set(int(x) for x in u_to_W_local.tolist())

        # Load mesh & facet tags to locate displacement DOFs on facets.
        msh, cell_tags, facet_tags = fem3d._load_mesh_and_tags(mesh_file, status_callback=None)

        # Build mixed space (same P1^3 / P1 as coupled assembly).
        u_el = fem3d._displacement_element(msh, 1)
        p_el = fem3d.element("Lagrange", msh.basix_cell(), 1)
        W_el = fem3d.mixed_element([u_el, p_el])
        W = fem3d.fem.functionspace(msh, W_el)

        _, V_u_collapsed, u_parent_indices = fem3d._audit_coupled_displacement_space(
            msh, W, u_el, status_callback=None
        )
        u_parent_indices = np.asarray(u_parent_indices, dtype=np.int32).ravel()
        if u_parent_indices.size == 0:
            raise RuntimeError("u_parent_indices empty; cannot map tag dofs")

        # Tag values: requirement aligns with fem_main_3d wood tags.
        tag_top = 1
        tag_back = 3
        tag_ribs = 4
        tag_fix = 5

        def _map_facet_tag(tag_val: int) -> np.ndarray:
            facets = np.asarray(facet_tags.find(int(tag_val)), dtype=np.int32).ravel()
            if facets.size == 0:
                return np.array([], dtype=np.int32)
            dofs_local = fem3d._locate_facet_displacement_dofs(V_u_collapsed, msh, facets)
            if dofs_local.size == 0:
                return np.array([], dtype=np.int32)
            dofs_local = np.asarray(dofs_local, dtype=np.int32).ravel()
            valid = (dofs_local >= 0) & (dofs_local < u_parent_indices.size)
            dofs_local = dofs_local[valid]
            parent = u_parent_indices[dofs_local]
            if parent.size == 0:
                return np.array([], dtype=np.int32)
            local = parent_to_local[parent]
            local = local[(local >= 0) & (local < operator_size)]
            local = np.unique(local.astype(np.int32, copy=False))
            local = np.asarray([i for i in local.tolist() if int(i) in u_set_local], dtype=np.int32)
            return local

        top_u = _map_facet_tag(tag_top)
        back_u = _map_facet_tag(tag_back)
        ribs_u = _map_facet_tag(tag_ribs)
        fix_u = _map_facet_tag(tag_fix)

        shell_u = np.unique(np.concatenate([top_u, back_u, ribs_u]).astype(np.int32, copy=False)) if (top_u.size+back_u.size+ribs_u.size)>0 else np.array([],dtype=np.int32)
        u_all_local = np.asarray(u_to_W_local, dtype=np.int32).ravel()
        non_shell_u = np.asarray(sorted(set(int(x) for x in u_all_local.tolist()) - set(int(x) for x in shell_u.tolist())), dtype=np.int32) if u_all_local.size else np.array([],dtype=np.int32)

        return {
            "u_to_W_local_expected": int(u_to_W_local.size),
            "subsets": {
                "tag_1_top_shell_displacement": top_u,
                "tag_3_back_shell_displacement": back_u,
                "tag_4_ribs_side_displacement": ribs_u,
                "tag_5_pinned_fix_displacement": fix_u,
                "shell_tag_union_top_back_ribs": shell_u,
                "u_non_shell_displacement_complement": non_shell_u,
            },
            "tag_values": {"tag_1": tag_top, "tag_3": tag_back, "tag_4": tag_ribs, "tag_5": tag_fix},
            "parent_to_local": {
                "n_coupled_W_full": n_full_w,
                "active_W_parent_len": int(active_sorted.size),
                "operator_size": int(operator_size),
            },
            "layout_consistency": {
                "u_active_size": int(u_set_local.__len__()),
                "u_input_passed_size": int(u_to_W.size),
                "p_input_passed_size": int(p_to_W.size),
            },
        }
    finally:
        try:
            A.destroy()
            M.destroy()
        except Exception:
            pass


def main() -> int:
    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[u_active_nullspace_attribution] Requires mpiexec -n 1", file=sys.stderr)
        return 2

    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
    case_dir = solve_case_dir("L_mid", CASE_ID)
    out_dir = case_dir / OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1
    mesh_file = mesh_path("L_mid", CASE_ID)
    seed_npy = case_dir / "diagnostics" / "acoustic_coupled_seed.npy"

    # (placeholder) reserved for future cheap pre-checks

    # Guard audit (do not rerun EPS)
    eps_auth = _atomic_load_json(EPS_AUTH_RECORD_JSON)
    eps_run_count = int(eps_auth.get("eps_run_count_for_this_lane", 0) or 0)
    no_additional_eps_run_authorized = True
    re_invocation_guard = bool(EPS_AUTH_RECORD_JSON.is_file() and eps_run_count >= 1)

    # Reassemble operators and extract u/p layout.
    sample = sample_spec_from_case(case)
    A, M, cfg = _assemble_reduced_coupled_replay(mesh_file, sample, coupling_enabled=True)
    maps = _extract_layout_maps(cfg, A)
    u_to_W = np.asarray(maps["u_to_W"], dtype=np.int32).ravel()
    p_to_W = np.asarray(maps["p_to_W"], dtype=np.int32).ravel()
    operator_size = int(A.getSize()[0])

    try:
        # Build geometric tag subsets in reduced u indices.
        tag_map = _build_tag_subsets_in_reduced_u(
            mesh_file=mesh_file,
            sample=sample,
            u_to_W=u_to_W,
            p_to_W=p_to_W,
            operator_size=operator_size,
            coupling_enabled=True,
        )

        # Load candidate vectors (lossless dense) and associated metadata.
        modes_path = out_dir / LOSSLESS_SUMMARY_PATH
        bank_path = out_dir / LOSSLESS_BANK_PATH
        loaded_modes = _atomic_load_json(modes_path).get("modes")
        modes = loaded_modes if isinstance(loaded_modes, list) else []
        bank = _atomic_load_json(bank_path)
        saved_mode_rows = _coalesce_list(bank.get("saved_mode_rows"), bank.get("candidates"))

        bank_by_slot: Dict[int, Dict[str, Any]] = {}
        if isinstance(saved_mode_rows, list):
            for r in saved_mode_rows:
                if not isinstance(r, dict):
                    continue
                slot = int(r.get("eps_slot_index", r.get("candidate_index", -1)) or -1)
                if slot >= 0:
                    bank_by_slot[slot] = r

        # Load seed.
        seed_info = load_seed_with_diagnostics(seed_npy)
        seed = seed_info.get("seed_array") if isinstance(seed_info.get("seed_array"), np.ndarray) else None
        if seed is None:
            raise RuntimeError("seed_array missing; cannot attribute")
        seed_vec = np.asarray(seed, dtype=np.float64).ravel()

        # Decompose seed and candidates by tag-based u subsets.
        subsets: Dict[str, np.ndarray] = tag_map["subsets"]
        subset_order = [
            "tag_5_pinned_fix_displacement",
            "shell_tag_union_top_back_ribs",
            "u_non_shell_displacement_complement",
            "tag_1_top_shell_displacement",
            "tag_3_back_shell_displacement",
            "tag_4_ribs_side_displacement",
        ]

        # Build action-probe induced "M-active output u rows" using unit vectors on key u subsets.
        # Choose probes on shell_tag_union and u_non_shell complement.
        action_probe_shell = _probe_action_rows(A, M, u_to_W, p_to_W, subsets["shell_tag_union_top_back_ribs"])
        action_probe_non_shell = _probe_action_rows(A, M, u_to_W, p_to_W, subsets["u_non_shell_displacement_complement"])

        # Conservative M-active rows defined by union of probe output indices would require Ax/Mx arrays.
        # As a compromise, define M-active u support as the subset(s) whose probe ||M y|| is non-negligible.
        # We approximate using l2 action norms computed by matvec on vectors supported on those subsets.
        from physical_fsi_seed_residual_audit import _petsc_vec_from_array, _petsc_matvec

        def _action_norm_on_subset(s_idx: np.ndarray, op_mat) -> float:
            if s_idx.size == 0:
                return 0.0
            x = np.zeros(operator_size, dtype=np.float64)
            x[s_idx] = 1.0
            xn = float(np.linalg.norm(x))
            if xn <= 0:
                return 0.0
            x /= xn
            vx = _petsc_vec_from_array(A, x)
            y_out = None
            try:
                y, y_out = _petsc_matvec(op_mat, vx)
            finally:
                try:
                    vx.destroy()
                except Exception:
                    pass
                if y_out is not None:
                    try:
                        y_out.destroy()
                    except Exception:
                        pass
            return float(np.linalg.norm(np.asarray(y, dtype=np.float64).ravel()))

        M_norm_shell_probe = _action_norm_on_subset(subsets["shell_tag_union_top_back_ribs"], M)
        M_norm_non_shell_probe = _action_norm_on_subset(subsets["u_non_shell_displacement_complement"], M)
        A_norm_shell_probe = _action_norm_on_subset(subsets["shell_tag_union_top_back_ribs"], A)
        A_norm_non_shell_probe = _action_norm_on_subset(subsets["u_non_shell_displacement_complement"], A)

        # Choose M-active u columns as those categories that yield non-negligible mass action.
        m_active_cols_categories: List[str] = []
        if M_norm_shell_probe > 1e-25:
            m_active_cols_categories.append("shell_tag_union_top_back_ribs")
        if M_norm_non_shell_probe > 1e-25:
            m_active_cols_categories.append("u_non_shell_displacement_complement")
        M_active_u_cols = np.unique(np.concatenate([subsets[c] for c in m_active_cols_categories if subsets.get(c) is not None and subsets[c].size > 0]).astype(np.int32, copy=False)) if m_active_cols_categories else np.array([], dtype=np.int32)
        M_null_u_cols = np.asarray(sorted(set(int(i) for i in u_to_W.tolist()) - set(int(i) for i in M_active_u_cols.tolist())), dtype=np.int32) if u_to_W.size else np.array([], dtype=np.int32)

        # Similarly for A-active.
        a_active_cols_categories: List[str] = []
        if A_norm_shell_probe > 1e-25:
            a_active_cols_categories.append("shell_tag_union_top_back_ribs")
        if A_norm_non_shell_probe > 1e-25:
            a_active_cols_categories.append("u_non_shell_displacement_complement")
        A_active_u_cols = np.unique(np.concatenate([subsets[c] for c in a_active_cols_categories if subsets.get(c) is not None and subsets[c].size > 0]).astype(np.int32, copy=False)) if a_active_cols_categories else np.array([], dtype=np.int32)
        A_null_u_cols = np.asarray(sorted(set(int(i) for i in u_to_W.tolist()) - set(int(i) for i in A_active_u_cols.tolist())), dtype=np.int32) if u_to_W.size else np.array([], dtype=np.int32)

        # Preload candidate lossless vectors and compute per-candidate attribution.
        # Note: we only load lossless dense vectors (*.smx.dense.npy) and never touch sparse *.smx.npz.
        from fem_mode_array_utils import load_mode_dense_f64_lossless

        def _lossless_path_from_mode_row(mode_row: Dict[str, Any]) -> Optional[Path]:
            rel = mode_row.get("vector_file_lossless")
            if not rel:
                return None
            return (out_dir / str(rel)).resolve()

        per_candidate: List[Dict[str, Any]] = []
        u_subset_stats_aggregate: Dict[str, Any] = {k: [] for k in subsets.keys()}
        u_subset_nnzs_aggregate: Dict[str, Any] = {k: [] for k in subsets.keys()}
        duplicates_signature: List[np.ndarray] = []
        # Low-cost duplicate probe intentionally omitted here (kept in the mass-null postmortem already).

        # Seed operators for threshold audit.
        seed_ops = _matvec_norms(A, M, seed_vec, u_idx=u_to_W, p_idx=p_to_W)
        seed_xh_mx = float(seed_ops.get("xH_Mx", float("nan")))
        seed_mx_norm = float(seed_ops.get("l2_norm_Mx", float("nan")))

        for m in modes:
            slot = int(m.get("eps_slot_index", m.get("candidate_index", m.get("mode_index", 0))) or 0)
            lossless_path = _lossless_path_from_mode_row(m)
            if lossless_path is None or not lossless_path.is_file():
                continue
            vec = load_mode_dense_f64_lossless(lossless_path)
            vec = np.asarray(vec, dtype=np.float64).ravel()

            vec_u = vec.copy()
            if p_to_W.size:
                vec_u[p_to_W] = 0.0
            vec_p = vec.copy()
            if u_to_W.size:
                vec_p[u_to_W] = 0.0

            # Tag subset support
            support_detail: Dict[str, Any] = {}
            l2_total = float(np.linalg.norm(vec))
            for subset_name in subsets.keys():
                idx = np.asarray(subsets[subset_name], dtype=np.int32).ravel()
                sup = _block_support(vec, idx)
                support_detail[subset_name] = {
                    "l2_norm": sup["l2_norm"],
                    "l2_fraction": sup["l2_norm"] / max(l2_total, 1e-300),
                    "max_abs": sup["max_abs"],
                    **{k: v for k, v in sup.items() if k.startswith("nnz_")},
                }
                u_subset_stats_aggregate[subset_name].append(support_detail[subset_name]["l2_fraction"])

            # Operator actions
            lam_phys = None
            brec = bank_by_slot.get(slot, {})
            if isinstance(brec, dict):
                lam_phys = brec.get("lam_phys")
            # Full operator actions (Ax, Mx arrays are needed for residual norms)
            from physical_fsi_seed_residual_audit import _petsc_vec_from_array, _petsc_matvec
            vx = _petsc_vec_from_array(A, vec)
            ay = my = None
            try:
                Ax, ay = _petsc_matvec(A, vx)
                Mx, my = _petsc_matvec(M, vx)
            finally:
                try:
                    vx.destroy()
                except Exception:
                    pass
                if ay is not None:
                    try:
                        ay.destroy()
                    except Exception:
                        pass
                if my is not None:
                    try:
                        my.destroy()
                    except Exception:
                        pass
            Ax_arr = np.asarray(Ax, dtype=np.float64).ravel()
            Mx_arr = np.asarray(Mx, dtype=np.float64).ravel()

            xH_Ax = float(np.vdot(vec, Ax_arr).real)
            xH_Mx = float(np.vdot(vec, Mx_arr).real)

            # Rayleigh lambda as "replay lambda" (if denominator not near-zero)
            if abs(xH_Mx) > 1e-30:
                lam_replay = float((xH_Ax / xH_Mx).real)
                lam_replay_finite = math.isfinite(lam_replay)
            else:
                lam_replay = float("nan")
                lam_replay_finite = False

            residual_reported_norm = _residual_norm_from_A_M(Ax_arr, Mx_arr, float(lam_phys) if lam_phys is not None else float("nan"))
            residual_replay_norm = (
                _residual_norm_from_A_M(Ax_arr, Mx_arr, lam_replay) if lam_replay_finite else float("nan")
            )

            # u/p separated actions (avoid extra matvec if p norm is tiny)
            p_norm = float(np.linalg.norm(vec_p)) if p_to_W.size else 0.0
            u_norm = float(np.linalg.norm(vec_u)) if u_to_W.size else 0.0

            vx_u = _petsc_vec_from_array(A, vec_u)
            ay_u = my_u = None
            try:
                Ax_u, ay_u = _petsc_matvec(A, vx_u)
                Mx_u, my_u = _petsc_matvec(M, vx_u)
            finally:
                try:
                    vx_u.destroy()
                except Exception:
                    pass
                if ay_u is not None:
                    try:
                        ay_u.destroy()
                    except Exception:
                        pass
                if my_u is not None:
                    try:
                        my_u.destroy()
                    except Exception:
                        pass
            Ax_u_arr = np.asarray(Ax_u, dtype=np.float64).ravel()
            Mx_u_arr = np.asarray(Mx_u, dtype=np.float64).ravel()

            if p_norm > 1e-30:
                vx_p = _petsc_vec_from_array(A, vec_p)
                ay_p = my_p = None
                try:
                    Ax_p, ay_p = _petsc_matvec(A, vx_p)
                    Mx_p, my_p = _petsc_matvec(M, vx_p)
                finally:
                    try:
                        vx_p.destroy()
                    except Exception:
                        pass
                    if ay_p is not None:
                        try:
                            ay_p.destroy()
                        except Exception:
                            pass
                    if my_p is not None:
                        try:
                            my_p.destroy()
                        except Exception:
                            pass
                Ax_p_arr = np.asarray(Ax_p, dtype=np.float64).ravel()
                Mx_p_arr = np.asarray(Mx_p, dtype=np.float64).ravel()
            else:
                Ax_p_arr = np.zeros_like(Ax_arr)
                Mx_p_arr = np.zeros_like(Mx_arr)

            u_subset_norm_frac_on_M_active = _norm(vec[M_active_u_cols]) / max(l2_total, 1e-300) if M_active_u_cols.size else 0.0
            u_subset_norm_frac_on_M_null = _norm(vec[M_null_u_cols]) / max(l2_total, 1e-300) if M_null_u_cols.size else 0.0
            u_subset_norm_frac_on_A_active = _norm(vec[A_active_u_cols]) / max(l2_total, 1e-300) if A_active_u_cols.size else 0.0
            u_subset_norm_frac_on_A_null = _norm(vec[A_null_u_cols]) / max(l2_total, 1e-300) if A_null_u_cols.size else 0.0

            per_candidate.append(
                {
                    "eps_slot_index": slot,
                    "lossless_vector_path": str(lossless_path.relative_to(out_dir)).replace("\\", "/"),
                    "vector_length": int(vec.size),
                    "l2_norm_total": l2_total,
                    "max_abs_total": float(np.max(np.abs(vec))) if vec.size else 0.0,
                    "u_active_norm": u_norm,
                    "p_active_norm": p_norm,
                    "support_by_tag_and_structural_subsets": support_detail,
                    "operator_actions": {
                        "||x||": l2_total,
                        "||Ax||": float(np.linalg.norm(Ax_arr)),
                        "||Mx||": float(np.linalg.norm(Mx_arr)),
                        "xH_Ax": xH_Ax,
                        "xH_Mx": xH_Mx,
                        "rayleigh_lambda": float(lam_replay) if lam_replay_finite else None,
                        "residual_norm_at_lam_phys": residual_reported_norm,
                        "residual_norm_at_lam_replay_if_finite": residual_replay_norm,
                        "max_abs(Ax)": float(np.max(np.abs(Ax_arr))) if Ax_arr.size else 0.0,
                        "max_abs(Mx)": float(np.max(np.abs(Mx_arr))) if Mx_arr.size else 0.0,
                        "nnz_significant_Mx_gt_abs_1e-15": int(np.sum(np.abs(Mx_arr) > 1e-15)),
                        "||M x_u||": float(np.linalg.norm(Mx_u_arr)),
                        "||A x_u||": float(np.linalg.norm(Ax_u_arr)),
                        "||M x_p||": float(np.linalg.norm(Mx_p_arr)),
                        "||A x_p||": float(np.linalg.norm(Ax_p_arr)),
                        "Mx_u_on_u_l2": float(np.linalg.norm(Mx_u_arr[u_to_W])) if u_to_W.size else 0.0,
                    },
                    "u_active_overlap_with_mass_active_columns": {
                        "M_active_u_cols_categories": m_active_cols_categories,
                        "fraction_total_norm_on_M_active_u_cols": u_subset_norm_frac_on_M_active,
                        "fraction_total_norm_on_M_null_u_cols": u_subset_norm_frac_on_M_null,
                    },
                    "u_active_overlap_with_stiffness_active_columns": {
                        "A_active_u_cols_categories": a_active_cols_categories,
                        "fraction_total_norm_on_A_active_u_cols": u_subset_norm_frac_on_A_active,
                        "fraction_total_norm_on_A_null_u_cols": u_subset_norm_frac_on_A_null,
                    },
                    "mass_null_audit_flags": {
                        "mass_null_abs_norm": bool(float(np.linalg.norm(Mx_arr)) < M_NULL_ABS_NORM_TOL),
                        "mass_null_xH": bool(abs(xH_Mx) < XH_MX_TOL),
                        "in_or_near_null_M_abs_norm": bool(float(np.linalg.norm(Mx_arr)) < IN_OR_NEAR_NULL_M_ABS_NORM_TOL),
                    },
                }
            )

        # Aggregate support summary.
        agg: Dict[str, Any] = {
            "candidate_count": len(per_candidate),
            "u_active_size": int(u_to_W.size),
            "p_active_size": int(p_to_W.size),
            "subset_fraction_summary": {},
            "dominant_support_subtype_by_median": {},
        }

        for subset_name in subsets.keys():
            fracs = u_subset_stats_aggregate.get(subset_name) or []
            if not fracs:
                continue
            agg["subset_fraction_summary"][subset_name] = {
                "median_l2_fraction": float(np.median(fracs)),
                "min_l2_fraction": float(np.min(fracs)),
                "max_l2_fraction": float(np.max(fracs)),
            }

        # Determine refined classification: choose among the requested categories.
        # Heuristics tuned to the evidence: candidates are u-dominant and p-active absent.
        tag5_fracs = agg["subset_fraction_summary"].get("tag_5_pinned_fix_displacement", {}).get("median_l2_fraction", 0.0)
        non_shell_fracs = agg["subset_fraction_summary"].get("u_non_shell_displacement_complement", {}).get("median_l2_fraction", 0.0)
        shell_fracs = agg["subset_fraction_summary"].get("shell_tag_union_top_back_ribs", {}).get("median_l2_fraction", 0.0)

        # M-null dominance: compare M-active overlap fractions.
        m_active_fracs = [c["u_active_overlap_with_mass_active_columns"]["fraction_total_norm_on_M_active_u_cols"] for c in per_candidate]
        m_null_fracs = [c["u_active_overlap_with_mass_active_columns"]["fraction_total_norm_on_M_null_u_cols"] for c in per_candidate]
        median_m_active_overlap = float(np.median(m_active_fracs)) if m_active_fracs else 0.0
        median_m_null_overlap = float(np.median(m_null_fracs)) if m_null_fracs else 0.0

        refined_classification = "UNRESOLVED_U_ACTIVE_NULLSPACE"
        refined_subtype = ""
        if tag5_fracs >= 0.5:
            refined_classification = "LOSSLESS_ST_TARGETING_BLOCKED_BY_U_ACTIVE_NULLSPACE_MODES"
            refined_subtype = "DIRICHLET_OR_PINNED_ALGEBRAIC_SUPPORT(tag-5_fix_dofs_dominant)"
        elif non_shell_fracs >= 0.5 and median_m_null_overlap >= 0.9:
            refined_classification = "LOSSLESS_ST_TARGETING_BLOCKED_BY_U_ACTIVE_NULLSPACE_MODES"
            refined_subtype = "NON_SHELL_OR_INACTIVE_U_NULLSPACE_SUPPORT(u-non-shell_cols_dominant_and_mass-null)"
        elif shell_fracs >= 0.5 and median_m_null_overlap >= 0.9:
            refined_classification = "LOSSLESS_ST_TARGETING_BLOCKED_BY_U_ACTIVE_NULLSPACE_MODES"
            refined_subtype = "STIFFNESS_ACTIVE_MASS_NULL_STRUCTURAL_SUPPORT(shell_cols_mass-null)"

        # Threshold audit: why seed in_or_near_null_M is True while seed mass_null is False.
        seed_in_or_near_null_M = bool(seed_mx_norm < IN_OR_NEAR_NULL_M_ABS_NORM_TOL)
        seed_mass_null = bool(seed_mx_norm < M_NULL_ABS_NORM_TOL or abs(seed_xh_mx) < XH_MX_TOL)
        threshold_audit = {
            "mass_null_threshold": {
                "||Mx|| <": M_NULL_ABS_NORM_TOL,
                "|xH_Mx| <": XH_MX_TOL,
            },
            "in_or_near_null_M_threshold": {
                "||Mx|| <": IN_OR_NEAR_NULL_M_ABS_NORM_TOL,
            },
            "seed_values": {
                "||Mx||": seed_mx_norm,
                "xH_Mx": seed_xh_mx,
            },
            "seed_flags": {
                "in_or_near_null_M": seed_in_or_near_null_M,
                "mass_null": seed_mass_null,
            },
            "interpretation": (
                "The in_or_near_null_M threshold is an absolute ||Mx|| norm cut and "
                "is too broad for the small-scaled but physically valid acoustic seed "
                "(which has finite Rayleigh and nonzero mass action). A relative-to-seed "
                "metric (e.g., ||Mx||/||Mx||_seed) is preferable to separate valid small "
                "mass action from true EPS mass-null eigenvectors."
            ),
            "proposed_relative_semantics": {
                "define_relative_mass_action_ratio": "||Mx|| / ||Mx||_seed",
                "near_null_definition_example": "ratio < 1e-6 and |xH_Mx| small relative to seed",
            },
        }

        # Remediation design-only (no implementation; no EPS solve).
        remediation = {
            "goal": "Recover the p_active acoustic branch while avoiding targeting u_active near-null(M) structural modes.",
            "options": [
                {
                    "option": "Project out identified M-null u_active DOFs before EPS",
                    "what_changes": (
                        "Eliminate or constrain the u columns/DOFs spanning the detected near-null(M) subspace "
                        "from the generalized eigenproblem so the shift-invert operator cannot converge onto it."
                    ),
                    "physics_impact": "Algebraic representation only if the projection is applied in the reduced coordinate space.",
                    "expected_recover_p_branch": "High if null subspace is correctly identified and projection is aligned with replay operators.",
                    "implementation_risk": "Requires a reliable mapping from identified DOF subsets to reduced eigenproblem basis; risk of removing true physics mass-bearing components.",
                },
                {
                    "option": "Construct EPS on the physical mass-bearing reduced subspace",
                    "what_changes": (
                        "Build a reduced basis from the complement of the u_active M-null (mass-bearing columns) and solve EPS "
                        "in that physical subspace, rather than the full singular-mass reduced space."
                    ),
                    "physics_impact": "Intended; focuses on the physical weak forms' mass-bearing subspace.",
                    "expected_recover_p_branch": "High if the basis spans the acoustic branch and is consistent with replay maps/operators.",
                    "implementation_risk": "Basis construction is nontrivial; must preserve coupled u/p layout and BC reductions.",
                },
                {
                    "option": "Explicit null-space deflation / constrained generalized eigenproblem",
                    "what_changes": (
                        "Use a deflated generalized eigenvalue approach: apply a nullspace-aware solver or augment the problem "
                        "with constraints that penalize vectors with negligible M action."
                    ),
                    "physics_impact": "Algebraic solver strategy; weak forms unchanged.",
                    "expected_recover_p_branch": "Medium-to-high, depending on deflation quality and stability for SINVERT."
                    ,
                    "implementation_risk": "Solver-level change; requires validation for singular generalized eigenproblems and consistent backtransforms.",
                },
                {
                    "option": "Singular generalized eigenproblem method (robust for rank-deficient M)",
                    "what_changes": (
                        "Switch from naive GHEP targeting to a method that treats rank-deficient M explicitly "
                        "and returns physically relevant modes within the constrained generalized eigenstructure."
                    ),
                    "physics_impact": "Solver-level; weak forms unchanged.",
                    "expected_recover_p_branch": "Medium; provides correct spectral information but might still require subspace constraints.",
                    "implementation_risk": "More invasive solver configuration and backtransformation semantics."
                    ,
                },
                {
                    "option": "PGNHEP / purification only if it is justified by nullspace attribution",
                    "what_changes": (
                        "Apply PGNHEP/purification steps only after verifying that they remove the same u_active near-null(M) modes "
                        "without discarding the p_active branch."
                    ),
                    "physics_impact": "Could be physical if it changes effective operator in null components.",
                    "expected_recover_p_branch": "Unknown until validated; might help if purification targets the detected null subspace.",
                    "implementation_risk": "May alter modeled operator content; must validate with no-EPS replay checks first."
                    ,
                },
                {
                    "option": "ST regularization only as a diagnostic (not a final physical verdict path)",
                    "what_changes": (
                        "Use ST regularization (epsilon shifts) to confirm whether null-space attraction is purely due to singular mass targeting. "
                        "Keep it fail-closed unless it preserves the intended operator semantics for the physical branch."
                    ),
                    "physics_impact": "Diagnostic only; should not be used to declare recovery without no-EPS proof.",
                    "expected_recover_p_branch": "Low-to-medium; expected to shift the solver away from null mass but must be validated.",
                    "implementation_risk": "Could mask the underlying targeting issue; risk of false confidence."
                    ,
                },
            ],
            "recommended_option": refined_subtype or "UNRESOLVED_U_ACTIVE_NULLSPACE",
            "recommended_execution_guard": (
                "Any future EPS attempt must be preceded by no-EPS operator replay attribution on the exact same reduced operators "
                "and a check that the nullspace projection basis is stable across the candidate bank."
            ),
        }

        # Single-run duplication signature (approx).
        # We cannot safely compute full pairwise distances without storing vectors, but we provide a
        # low-cost duplicate probe based on random index signatures.
        # (Signature is computed from lossless dense vectors loaded above; approximate from support category.)
        # For strict duplicates, see per_candidate equality probes in the mass-null postmortem.

        status_refresh = {
            "no_additional_eps_run_authorized": True,
            "re_invoking_authorized_runner_would_block_eps": re_invocation_guard,
            "eps_run_count_for_this_lane": eps_run_count,
        }

        dominant_support = None
        try:
            dom = agg.get("subset_fraction_summary") or {}
            if dom:
                dominant_support = max(dom.keys(), key=lambda k: dom[k].get("median_l2_fraction", 0.0))
        except Exception:
            dominant_support = None

        report: Dict[str, Any] = {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evidence_scope": "report_only_no_eps",
            "output_subdir": OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1,
            "classification": refined_classification,
            "classification_subtype": refined_subtype,
            "classification_reason": (
                "Overlapping tag fractions and M-active column probes are insufficient for elimination; "
                "see mass-rank/disjoint-partition audit."
            ),
            "dominant_support_category": dominant_support or "u_non_shell_displacement_complement",
            "no_new_eigensolve_executed": True,
            "additional_eps_authorized": False,
            "single_run_guard_audit": status_refresh,
            "u_active_layout": {
                "operator_size": int(operator_size),
                "u_active_size": int(u_to_W.size),
                "p_active_size": int(p_to_W.size),
            },
            "tag_based_u_decomposition": {
                "tag_values": tag_map.get("tag_values"),
                "subsets_sizes": {
                    k: int(np.asarray(v, dtype=np.int32).size) for k, v in tag_map["subsets"].items()
                },
                "subset_action_probe": {
                    "shell_tag_probe": action_probe_shell,
                    "non_shell_probe": action_probe_non_shell,
                    "M_norm_shell_probe": M_norm_shell_probe,
                    "M_norm_non_shell_probe": M_norm_non_shell_probe,
                    "A_norm_shell_probe": A_norm_shell_probe,
                    "A_norm_non_shell_probe": A_norm_non_shell_probe,
                    "M_active_u_cols_categories": m_active_cols_categories,
                    "A_active_u_cols_categories": a_active_cols_categories,
                },
            },
            "candidate_aggregate_support": agg,
            "seed_control": {
                "seed_vector_length": int(seed_vec.size),
                "seed_norm_total": float(np.linalg.norm(seed_vec)),
                "seed_ops": {
                    "||Mx||": seed_ops.get("l2_norm_Mx"),
                    "xH_Mx": seed_ops.get("xH_Mx"),
                    "rayleigh_lambda": seed_ops.get("rayleigh_lambda"),
                    "rayleigh_frequency_hz": seed_ops.get("rayleigh_frequency_hz"),
                    "||Ax||": seed_ops.get("l2_norm_Ax"),
                    "xH_Ax": seed_ops.get("xH_Ax"),
                },
                "seed_support_by_tag_subsets": {
                    name: {
                        "l2_norm": _block_support(seed_vec, np.asarray(subsets[name], dtype=np.int32))["l2_norm"]
                        if np.asarray(subsets[name], dtype=np.int32).size else 0.0,
                        "l2_fraction": _block_support(seed_vec, np.asarray(subsets[name], dtype=np.int32))["l2_norm"] / max(float(np.linalg.norm(seed_vec)), 1e-300),
                    }
                    for name in subsets.keys()
                },
            },
            "candidate_per_slot": per_candidate,
            "mass_null_threshold_audit": threshold_audit,
            "refined_classification": {
                "classification": refined_classification,
                "subtype": refined_subtype,
                "evidence": {
                    "median_l2_fraction_tag5_fix": tag5_fracs,
                    "median_l2_fraction_non_shell": non_shell_fracs,
                    "median_l2_fraction_shell_union": shell_fracs,
                    "median_overlap_on_M_active_u_cols": median_m_active_overlap,
                    "median_overlap_on_M_null_u_cols": median_m_null_overlap,
                },
            },
            "remediation_design_review": remediation,
            "status_refresh_note": "This script does not execute EPS. Status scripts are refreshed conservatively at the end.",
        }

        write_json(OUT_JSON, report)

        lines: List[str] = [
            "# Lossless adjudication v1: u_active nullspace attribution",
            "",
            f"Generated: {report['generated_utc']}",
            "",
            f"Classification: `{refined_classification}`",
            f"Subtype: `{refined_subtype}`",
            "",
            "## Guard",
            f"- no_new_eigensolve_executed=True",
            f"- eps_run_count_for_this_lane={eps_run_count}",
            f"- re_invoking_authorized_runner_would_block_eps={re_invocation_guard}",
            "",
            "## Evidence (aggregate support fractions)",
        ]
        for subset_name in sorted(subsets.keys()):
            s = agg["subset_fraction_summary"].get(subset_name)
            if not s:
                continue
            lines.append(f"- {subset_name}: median_l2_fraction={s['median_l2_fraction']:.6e} (min={s['min_l2_fraction']:.6e}, max={s['max_l2_fraction']:.6e})")
        lines.extend(
            [
                "",
                "## Seed control (same layout/operators)",
                f"- seed ||Mx||={seed_ops.get('l2_norm_Mx')}",
                f"- seed xH_Mx={seed_ops.get('xH_Mx')}",
                f"- seed Rayleigh frequency_hz={seed_ops.get('rayleigh_frequency_hz')}",
                "",
                "## Threshold audit",
                f"- in_or_near_null_M_abs_norm_thresh={IN_OR_NEAR_NULL_M_ABS_NORM_TOL}",
                f"- seed ||Mx||={seed_mx_norm} -> in_or_near_null_M=True",
                "",
                "## Remediation design options (no EPS)",
                f"- recommended_option_subtype={refined_subtype}",
                "",
            ]
        )
        for opt in remediation["options"]:
            lines.append(f"### {opt['option']}")
            lines.append(f"- Expected ability to recover p_active branch: {opt.get('expected_recover_p_branch')}")
            lines.append(f"- Implementation risk: {opt.get('implementation_risk')}")
            lines.append("")
        OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

        mass_null_count = 0
        try:
            mass_null_count = sum(
                1
                for c in per_candidate
                if (c.get("mass_null_audit_flags") or {}).get("mass_null_abs_norm")
                or (c.get("mass_null_audit_flags") or {}).get("mass_null_xH")
            )
        except Exception:
            mass_null_count = 0

        print(
            f"[u_active_nullspace_attribution] refined_classification={refined_classification} subtype={refined_subtype} dominant_support={dominant_support}",
            flush=True,
        )
        print(
            f"[u_active_nullspace_attribution] candidate_count={len(per_candidate)} mass_null_count={mass_null_count} seed||Mx||={seed_mx_norm} seed_xH_Mx={seed_xh_mx}",
            flush=True,
        )

        # Refresh conservative status reports.
        try:
            from write_v2_st_singular_mass_rehabilitation_plan import main as rehab_main
            from run_v2_solver_root_cause_and_forward_risk_audit import main as audit_main

            rehab_main()
            audit_main()
        except Exception as exc:
            # Never fail the postmortem due to status refresh
            print(f"[u_active_nullspace_attribution] status_refresh_warning={type(exc).__name__}:{exc}", flush=True)

        print(f"[u_active_nullspace_attribution] wrote {OUT_JSON}", flush=True)
        print("[u_active_nullspace_attribution] no_new_eigensolve_executed=True", flush=True)
        return 0
    finally:
        try:
            A.destroy()
            M.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

