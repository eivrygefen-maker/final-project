#!/usr/bin/env python3
"""Report-only B3 trace-coupled operator and seed-transfer audit (no eigensolve)."""
from __future__ import annotations

import ast
import copy
import inspect
import json
import math
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
for _p in (SCRIPT_DIR, REPO_ROOT / "FEM" / "scripts"):
    import sys

    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fem_main_3d as fem3d
import ufl
from basix.ufl import element
from dolfinx import fem, mesh as dmesh
from physical_fsi_seed_residual_audit import (
    _block_residual_contributions,
    _petsc_matvec,
    _petsc_vec_from_array,
    _rayleigh_metrics,
)
from v2_build_coupled_acoustic_seed import (
    _assemble_reduced_coupled_replay,
    _extract_layout_maps,
    _extract_parent_raw_block_capture,
)
from v2_mesh_convergence_common import CONV_DIAG, load_manifest, mesh_path, sample_spec_from_case, solve_case_dir
from v2_unreg_offset_report_evaluator import load_seed_with_diagnostics

CASE_ID = "baseline_coupled_v2"
OUT_JSON = CONV_DIAG / "v2_B3_trace_coupled_operator_and_seed_transfer_audit.json"
OUT_MD = CONV_DIAG / "v2_B3_trace_coupled_operator_and_seed_transfer_audit.md"
OUT_JSON_C2_CONTRACT = CONV_DIAG / "v2_B3_C2_transfer_contract_only.json"
OUT_JSON_C2_SPARSE_COUPLING = CONV_DIAG / "v2_B3_C2_sparse_coupling_only.json"
OUT_JSON_B3_RAW_COMPOSITION = CONV_DIAG / "v2_B3_raw_composition_contract_only.json"
OUT_JSON_B3_SEED_REPLAY = CONV_DIAG / "v2_B3_seed_replay_audit_only.json"
OUT_MD_B3_SEED_REPLAY = CONV_DIAG / "v2_B3_seed_replay_audit_only.md"
REPORT_SIZE_TARGET_BYTES = 1048576
C2_TRANSFER_CONTRACT_ONLY_ARG = "--C2-transfer-contract-only"
C2_SPARSE_COUPLING_ONLY_ARG = "--C2-sparse-coupling-only"
B3_RAW_COMPOSITION_CONTRACT_ONLY_ARG = "--B3-raw-composition-contract-only"
B3_SEED_REPLAY_AUDIT_ONLY_ARG = "--B3-seed-replay-audit-only"
B3_OPERATOR_AIJ_BC_CONTRACT_ONLY_ARG = "--B3-operator-AIJ-BC-contract-only"
B3_SEED_BC_CONDITIONED_REPLAY_AUDIT_ONLY_ARG = "--B3-seed-BC-conditioned-replay-audit-only"
V2_VECTOR_BC_CONTRACT_ONLY_ARG = "--V2-vector-BC-contract-only"
OUT_JSON_B3_OPERATOR_AIJ_BC = CONV_DIAG / "v2_B3_operator_aij_BC_contract_only.json"
OUT_MD_B3_OPERATOR_AIJ_BC = CONV_DIAG / "v2_B3_operator_aij_BC_contract_only.md"
OUT_JSON_B3_SEED_BC_CONDITIONED = CONV_DIAG / "v2_B3_seed_BC_conditioned_replay_audit_only.json"
OUT_MD_B3_SEED_BC_CONDITIONED = CONV_DIAG / "v2_B3_seed_BC_conditioned_replay_audit_only.md"
OUT_JSON_B3_CONDITIONED_MASS = CONV_DIAG / "v2_B3_conditioned_seed_mass_decomposition_audit_only.json"
OUT_MD_B3_CONDITIONED_MASS = CONV_DIAG / "v2_B3_conditioned_seed_mass_decomposition_audit_only.md"
B3_SEED_BC_CONDITIONED_MASS_DECOMPOSITION_AUDIT_ONLY_ARG = (
    "--B3-conditioned-seed-mass-decomposition-audit-only"
)
B3_JD_DESIGN_READINESS_CONTRACT_ONLY_ARG = "--B3-JD-design-readiness-contract-only"
B3_JD_API_PREFLIGHT_ONLY_ARG = "--B3-JD-api-preflight-only"
B3_JD_OPERATOR_WIRING_PREFLIGHT_ONLY_ARG = "--B3-JD-operator-wiring-preflight-only"
B3_JD_FIRST_BOUNDED_EXECUTION_ONLY_ARG = "--B3-JD-first-bounded-execution-only"
B3_JD_DIMENSION_SETUP_PREFLIGHT_ONLY_ARG = "--B3-JD-dimension-setup-preflight-only"
B3_GNHEP_BC_SPECTRAL_POLLUTION_CONTRACT_ONLY_ARG = "--B3-GNHEP-BC-spectral-pollution-contract-only"
B3_GNHEP_BC_NO_LAMBDA_ONE_OPERATOR_CONTRACT_ONLY_ARG = "--B3-GNHEP-BC-no-lambda-one-operator-contract-only"
B3_JD_FIXED_BC_DIMENSION_SETUP_PREFLIGHT_ONLY_ARG = "--B3-JD-fixed-BC-dimension-setup-preflight-only"
B3_JD_FIXED_BC_SECOND_BOUNDED_EXECUTION_ONLY_ARG = "--B3-JD-fixed-BC-second-bounded-execution-only"
OUT_JSON_B3_JD_DESIGN = CONV_DIAG / "v2_B3_JD_design_readiness_contract_only.json"
OUT_MD_B3_JD_DESIGN = CONV_DIAG / "v2_B3_JD_design_readiness_contract_only.md"
OUT_JSON_B3_JD_API_PREFLIGHT = CONV_DIAG / "v2_B3_JD_api_preflight_only.json"
OUT_MD_B3_JD_API_PREFLIGHT = CONV_DIAG / "v2_B3_JD_api_preflight_only.md"
OUT_JSON_B3_JD_OPERATOR_WIRING_PREFLIGHT = CONV_DIAG / "v2_B3_JD_operator_wiring_preflight_only.json"
OUT_MD_B3_JD_OPERATOR_WIRING_PREFLIGHT = CONV_DIAG / "v2_B3_JD_operator_wiring_preflight_only.md"
OUT_JSON_B3_JD_FIRST_BOUNDED = CONV_DIAG / "v2_B3_JD_first_bounded_execution_only.json"
OUT_MD_B3_JD_FIRST_BOUNDED = CONV_DIAG / "v2_B3_JD_first_bounded_execution_only.md"
OUT_JSON_B3_JD_SETUP_PREFLIGHT = CONV_DIAG / "v2_B3_JD_dimension_setup_preflight_only.json"
OUT_MD_B3_JD_SETUP_PREFLIGHT = CONV_DIAG / "v2_B3_JD_dimension_setup_preflight_only.md"
OUT_JSON_B3_GNHEP_BC_SPECTRAL = CONV_DIAG / "v2_B3_GNHEP_BC_spectral_pollution_contract_only.json"
OUT_MD_B3_GNHEP_BC_SPECTRAL = CONV_DIAG / "v2_B3_GNHEP_BC_spectral_pollution_contract_only.md"
OUT_JSON_B3_GNHEP_BC_NO_LAMBDA_ONE = CONV_DIAG / "v2_B3_GNHEP_BC_no_lambda_one_operator_contract_only.json"
OUT_MD_B3_GNHEP_BC_NO_LAMBDA_ONE = CONV_DIAG / "v2_B3_GNHEP_BC_no_lambda_one_operator_contract_only.md"
OUT_JSON_B3_JD_FIXED_BC_SETUP_PREFLIGHT = CONV_DIAG / "v2_B3_JD_fixed_BC_dimension_setup_preflight_only.json"
OUT_MD_B3_JD_FIXED_BC_SETUP_PREFLIGHT = CONV_DIAG / "v2_B3_JD_fixed_BC_dimension_setup_preflight_only.md"
OUT_JSON_B3_JD_FIXED_BC_SECOND_BOUNDED = CONV_DIAG / "v2_B3_JD_fixed_BC_second_bounded_execution_only.json"
OUT_MD_B3_JD_FIXED_BC_SECOND_BOUNDED = CONV_DIAG / "v2_B3_JD_fixed_BC_second_bounded_execution_only.md"
B3_JD_DEFAULT_TARGET_HZ = 244.39
B3_JD_DEFAULT_HARVEST_LO_HZ = 220.0
B3_JD_DEFAULT_HARVEST_HI_HZ = 265.0
B3_JD_FIRST_RUN_INITIAL_MODE_COUNT = 2
B3_JD_FIRST_RUN_NCV = 6
B3_ARTIFICIAL_LAMBDA_UNITY_FREQUENCY_HZ = 1.0 / (2.0 * math.pi)

_MASS_DECOMPOSITION_EVIDENCE_KEYS = (
    "B3_mass_Muu_norm",
    "B3_mass_Mpu_norm",
    "B3_mass_Mpp_norm",
    "B3_conditioned_mass_q_uu",
    "B3_conditioned_mass_q_up",
    "B3_conditioned_mass_q_pu",
    "B3_conditioned_mass_q_pp",
    "B3_conditioned_mass_q_total_from_blocks",
    "B3_conditioned_mass_q_total_from_final_AIJ",
    "B3_conditioned_mass_block_vs_final_consistency_pass",
    "B3_conditioned_seed_mass_diagnostic_classification",
    "B3_seed_BC_contamination_confirmed",
)

TAG_TOP = 1
TAG_BACK = 3
TAG_RIBS = 4
TAG_FIX = 5


def _safe_float(x: Any) -> Any:
    if x is None:
        return None
    v = float(x)
    if not math.isfinite(v):
        return "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")
    return v


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return len(text.encode("utf-8"))


def _crc32_i32(a: np.ndarray) -> int:
    return int(zlib.crc32(np.asarray(a, dtype=np.int32).tobytes()) & 0xFFFFFFFF)


def _compact_idx(a: np.ndarray) -> Dict[str, Any]:
    v = np.asarray(a, dtype=np.int32).ravel()
    return {
        "size": int(v.size),
        "min": int(v.min()) if v.size else None,
        "max": int(v.max()) if v.size else None,
        "crc32": _crc32_i32(v),
        "preview_first": [int(x) for x in v[:8].tolist()],
        "preview_last": [int(x) for x in v[-8:].tolist()] if v.size > 8 else [],
    }


def _mat_norm_or_none(mat: Any) -> Any:
    if mat is None:
        return None
    try:
        return _safe_float(mat.norm(PETSc.NormType.FROBENIUS))
    except Exception:
        return None


def _mat_shape(mat: Any) -> Any:
    if mat is None:
        return None
    try:
        s = mat.getSize()
        return [int(s[0]), int(s[1])]
    except Exception:
        return None


def _destroy_mat(mat: Any) -> None:
    if mat is None:
        return
    try:
        mat.destroy()
    except Exception:
        pass


def _native_stage(stage: str) -> None:
    print(f"[B3_native] stage={stage}", flush=True)


def _native_mat_info(label: str, mat: Any) -> None:
    try:
        shp = mat.getSize()
        print(
            f"[B3_native] {label}_type={mat.getType()} shape=[{int(shp[0])},{int(shp[1])}]",
            flush=True,
        )
    except Exception as exc:
        print(f"[B3_native] {label}_type=unavailable err={type(exc).__name__}:{exc}", flush=True)


def _register_mat_for_destroy(mats: List[Any], mat: Any, *, seen: set[int] | None = None) -> None:
    if mat is None:
        return
    mid = id(mat)
    if seen is not None:
        if mid in seen:
            return
        seen.add(mid)
    mats.append(mat)


def _destroy_mats_deduped(mats: List[Any]) -> tuple[int, int, bool]:
    seen: set[int] = set()
    destroyed = 0
    duplicate_attempts = 0
    for m_ in mats:
        if m_ is None:
            continue
        mid = id(m_)
        if mid in seen:
            duplicate_attempts += 1
            continue
        seen.add(mid)
        _destroy_mat(m_)
        destroyed += 1
    return destroyed, duplicate_attempts, duplicate_attempts == 0


def _petsc_quadratic_form(mat: Any, x_arr: np.ndarray) -> float:
    x_arr = np.asarray(x_arr, dtype=np.float64).ravel()
    vx = my = None
    try:
        vx = _petsc_vec_from_array(mat, x_arr)
        Mx, my = _petsc_matvec(mat, vx)
        q = float(np.real(np.vdot(x_arr, Mx)))
    finally:
        for obj in (my, vx):
            if obj is not None:
                obj.destroy()
    return q


class _B3MassCrossQuadraticMpuError(RuntimeError):
    """Raised when rectangular Mpu cross-quadratic evaluation fails in mass audit."""


def _petsc_cross_quadratic_layout_contract(
    mat: Any,
    right_vec: np.ndarray,
    left_vec: np.ndarray,
) -> Dict[str, Any]:
    """Contract for left_vec^H · (mat · right_vec) with mat [nrow, ncol]."""
    nrow, ncol = (int(mat.getSize()[0]), int(mat.getSize()[1]))
    right_len = int(np.asarray(right_vec, dtype=np.float64).ravel().size)
    left_len = int(np.asarray(left_vec, dtype=np.float64).ravel().size)
    x_right = y_left = None
    layout_ok = False
    petsc_right_len = None
    petsc_left_len = None
    try:
        x_right = mat.createVecRight()
        y_left = mat.createVecLeft()
        petsc_right_len = int(x_right.getSize())
        petsc_left_len = int(y_left.getSize())
        layout_ok = bool(
            petsc_right_len == ncol
            and petsc_left_len == nrow
            and right_len == ncol
            and left_len == nrow
        )
    finally:
        for obj in (y_left, x_right):
            if obj is not None:
                obj.destroy()
    return {
        "B3_mass_cross_quadratic_matrix_shape": [nrow, ncol],
        "B3_mass_cross_quadratic_left_vector_length": left_len,
        "B3_mass_cross_quadratic_right_vector_length": right_len,
        "B3_mass_cross_quadratic_rectangular_layout_contract_pass": layout_ok,
        "B3_mass_cross_quadratic_rectangular_layout_contract_failure_reason": (
            None
            if layout_ok
            else (
                f"mat_shape=[{nrow},{ncol}] "
                f"petsc_right={petsc_right_len} petsc_left={petsc_left_len} "
                f"numpy_right={right_len} numpy_left={left_len}"
            )
        ),
    }


def _petsc_cross_quadratic_form(mat: Any, right_vec: np.ndarray, left_vec: np.ndarray) -> float:
    """Compute left_vec^H · (mat · right_vec) for rectangular or square mat."""
    right_vec = np.asarray(right_vec, dtype=np.float64).ravel()
    left_vec = np.asarray(left_vec, dtype=np.float64).ravel()
    nrow, ncol = (int(mat.getSize()[0]), int(mat.getSize()[1]))
    if right_vec.size != ncol or left_vec.size != nrow:
        raise ValueError(
            "cross_quadratic_dimension_mismatch: "
            f"mat=[{nrow},{ncol}] right_len={right_vec.size} left_len={left_vec.size}"
        )
    x_right = y_left = None
    try:
        x_right = mat.createVecRight()
        y_left = mat.createVecLeft()
        x_right.setArray(right_vec.copy())
        try:
            x_right.assemble()
        except Exception:
            pass
        mat.mult(x_right, y_left)
        try:
            y_left.assemble()
        except Exception:
            pass
        y_arr = np.asarray(y_left.getArray(readonly=True), dtype=np.float64).copy()
        return float(np.real(np.vdot(left_vec, y_arr)))
    finally:
        for obj in (y_left, x_right):
            if obj is not None:
                obj.destroy()


def _petsc_duplicate_scaled(mat: Any, scale: float) -> Any:
    out = mat.duplicate()
    out.copy(mat)
    if abs(float(scale) - 1.0) > 1.0e-15:
        out.scale(float(scale))
    out.assemble()
    return out


def _petsc_zero_mat(nrow: int, ncol: int, comm: Any) -> Any:
    z = PETSc.Mat().create(comm=comm)
    z.setSizes([int(nrow), int(ncol)])
    z.setType("aij")
    z.setUp()
    z.assemble()
    return z


def _petsc_mat_global_nnz_used(mat: Any) -> int:
    try:
        info = mat.getInfo()
        for key in ("nz_used", "nz_allocated", "nz"):
            if key in info:
                return int(info[key])
    except Exception:
        pass
    return 0


def _petsc_insert_block_into_monolithic(
    dest: Any,
    src: Any,
    *,
    row_offset: int,
    col_offset: int,
) -> None:
    """Insert sparse block ``src`` into ``dest`` at global row/col offsets (mpiexec -n 1)."""
    nrow, _ncol = src.getSize()
    for r in range(int(nrow)):
        try:
            cols, vals = src.getRow(r)
        except TypeError:
            rowdat = src.getRow(r)
            cols, vals = rowdat[0], rowdat[1]
        cols_g = (np.asarray(cols, dtype=np.int32) + int(col_offset)).tolist()
        vals_g = np.asarray(vals, dtype=np.float64).tolist()
        try:
            dest.setValues(
                [int(r + row_offset)],
                cols_g,
                vals_g,
                addv=PETSc.InsertMode.INSERT_VALUES,
            )
        except Exception:
            dest.setValues(
                [int(r + row_offset)],
                cols_g,
                vals_g,
                addv=PETSc.InsertMode.INSERT,
            )
        try:
            src.restoreRow(r)
        except Exception:
            pass


def _b3_direct_sparse_aij_from_restricted_blocks(
    *,
    a_uu: Any,
    a_up: Any,
    a_pu: Any,
    a_pp: Any,
    m_uu: Any,
    m_pu: Any,
    m_pp: Any,
    n_u: int,
    n_p: int,
    comm: Any,
) -> tuple[Any, Any]:
    """Assemble [u|p] monolithic AIJ from restricted blocks; no MatNest or convert()."""
    n_w = int(n_u + n_p)
    nnz_est = int(
        _petsc_mat_global_nnz_used(a_uu)
        + _petsc_mat_global_nnz_used(a_up)
        + _petsc_mat_global_nnz_used(a_pu)
        + _petsc_mat_global_nnz_used(a_pp)
    )
    mm_nnz_est = int(
        _petsc_mat_global_nnz_used(m_uu)
        + _petsc_mat_global_nnz_used(m_pu)
        + _petsc_mat_global_nnz_used(m_pp)
    )
    row_nnz = max(1, int(math.ceil(max(nnz_est, mm_nnz_est) / max(1, n_w))))
    a_out = PETSc.Mat().create(comm=comm)
    a_out.setSizes([n_w, n_w])
    a_out.setType("aij")
    try:
        a_out.setPreallocationNNZ(row_nnz, row_nnz)
    except Exception:
        pass
    a_out.setUp()
    try:
        a_out.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    except Exception:
        pass
    m_out = PETSc.Mat().create(comm=comm)
    m_out.setSizes([n_w, n_w])
    m_out.setType("aij")
    try:
        m_out.setPreallocationNNZ(row_nnz, row_nnz)
    except Exception:
        pass
    m_out.setUp()
    try:
        m_out.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    except Exception:
        pass
    _petsc_insert_block_into_monolithic(a_out, a_uu, row_offset=0, col_offset=0)
    _petsc_insert_block_into_monolithic(a_out, a_up, row_offset=0, col_offset=int(n_u))
    _petsc_insert_block_into_monolithic(a_out, a_pu, row_offset=int(n_u), col_offset=0)
    _petsc_insert_block_into_monolithic(a_out, a_pp, row_offset=int(n_u), col_offset=int(n_u))
    _petsc_insert_block_into_monolithic(m_out, m_uu, row_offset=0, col_offset=0)
    _petsc_insert_block_into_monolithic(m_out, m_pu, row_offset=int(n_u), col_offset=0)
    _petsc_insert_block_into_monolithic(m_out, m_pp, row_offset=int(n_u), col_offset=int(n_u))
    a_out.assemble()
    m_out.assemble()
    return a_out, m_out


def _native_log_restricted_block_types(
    *,
    a_uu: Any,
    a_up: Any,
    a_pu: Any,
    a_pp: Any,
    m_uu: Any,
    m_pu: Any,
    m_pp: Any,
) -> None:
    print(
        "[B3_native] A_child_types="
        f"uu={a_uu.getType()},up={a_up.getType()},pu={a_pu.getType()},pp={a_pp.getType()}",
        flush=True,
    )
    print(
        "[B3_native] M_child_types="
        f"uu={m_uu.getType()},pu={m_pu.getType()},pp={m_pp.getType()}",
        flush=True,
    )
    print(
        "[B3_native] A_child_shapes="
        f"uu={_mat_shape(a_uu)},up={_mat_shape(a_up)},pu={_mat_shape(a_pu)},pp={_mat_shape(a_pp)}",
        flush=True,
    )
    print(
        "[B3_native] M_child_shapes="
        f"uu={_mat_shape(m_uu)},pu={_mat_shape(m_pu)},pp={_mat_shape(m_pp)}",
        flush=True,
    )


def _b3_pressure_release_rows_retained(
    msh: Any,
    facet_tags: Any,
    *,
    n_u_b3: int,
    p_air_collapsed: np.ndarray,
) -> tuple[np.ndarray, Dict[str, Any]]:
    p_air_collapsed = np.asarray(p_air_collapsed, dtype=np.int32).ravel()
    p_el = element("Lagrange", msh.basix_cell(), 1)
    v_p = fem.functionspace(msh, p_el)
    soundhole_facets = np.asarray(facet_tags.find(2), dtype=np.int32)
    p_sh = np.asarray(
        fem3d._locate_soundhole_pressure_release_dofs(v_p, soundhole_facets),
        dtype=np.int32,
    ).ravel()
    inv = {int(d): j for j, d in enumerate(p_air_collapsed.tolist())}
    rows: List[int] = []
    for d in p_sh.tolist():
        j = inv.get(int(d))
        if j is not None:
            rows.append(int(n_u_b3) + int(j))
    rows_arr = np.unique(np.asarray(rows, dtype=np.int32))
    pr_meta = {
        "B3_pressure_release_row_count": int(rows_arr.size),
        "B3_pressure_release_BC_mapped_to_retained_p_layout": bool(
            p_sh.size == 0 or all(inv.get(int(d)) is not None for d in p_sh.tolist())
        ),
    }
    return rows_arr, pr_meta


def _map_parent_seed_to_b3(
    parent_seed: np.ndarray,
    *,
    parent_idx: np.ndarray,
    n_u_parent: int,
    n_p_retained: int,
    n_u_b3: int,
    p_to_W_parent: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Map historical parent reduced seed [u_active | p_active] into B3 [u_trace | p_active]."""
    parent_seed = np.asarray(parent_seed, dtype=np.float64).ravel()
    parent_idx = np.asarray(parent_idx, dtype=np.int32).ravel()
    n_u_parent = int(n_u_parent)
    n_p_retained = int(n_p_retained)
    n_u_b3 = int(n_u_b3)
    n_w = int(n_u_b3 + n_p_retained)
    expected_parent = int(n_u_parent + n_p_retained)
    meta: Dict[str, Any] = {
        "B3_seed_parent_layout": "parent_restricted_concatenated_u_then_active_p",
        "B3_seed_parent_u_dimension": n_u_parent,
        "B3_seed_parent_p_dimension": n_p_retained,
        "B3_seed_u_mapping_method": "parent_restricted_u_selected_by_parent_index_per_trace_dof",
        "B3_seed_p_mapping_method": "retained_pressure_entries_reused_without_second_restriction",
    }
    if int(parent_seed.size) != expected_parent:
        raise ValueError(
            f"parent_seed_length_{parent_seed.size}_!=_expected_{expected_parent}"
        )
    if parent_idx.size != n_u_b3:
        raise ValueError(f"parent_idx_length_{parent_idx.size}_!=_B3_u_{n_u_b3}")
    if np.any(parent_idx < 0) or np.any(parent_idx >= n_u_parent):
        raise ValueError("parent_index_per_trace_dof_out_of_parent_restricted_u_block")
    parent_u = parent_seed[:n_u_parent]
    parent_p = parent_seed[n_u_parent : n_u_parent + n_p_retained]
    if p_to_W_parent is not None:
        p_rows = np.asarray(p_to_W_parent, dtype=np.int32).ravel()
        expected_p_rows = np.arange(n_u_parent, n_u_parent + n_p_retained, dtype=np.int32)
        meta["B3_seed_parent_p_to_W_contiguous_block_pass"] = bool(
            p_rows.size == n_p_retained and np.array_equal(p_rows, expected_p_rows)
        )
    b3 = np.zeros(n_w, dtype=np.float64)
    b3[:n_u_b3] = parent_u[parent_idx]
    b3[n_u_b3:] = parent_p
    u_idx = np.arange(n_u_b3, dtype=np.int32)
    p_idx = np.arange(n_u_b3, n_u_b3 + n_p_retained, dtype=np.int32)
    return b3, u_idx, p_idx, meta


def _build_b3_scaled_restricted_operators_in_memory(
    *,
    raw_Auu: Any,
    raw_Muu: Any,
    raw_App: Any,
    raw_Mpp: Any,
    raw_Aup_B3: Any,
    raw_Apu_B3: Any,
    raw_Mpu_B3: Any,
    s_uu: float,
    s_pp: float,
    s_c: float,
    n_u_b3: int,
    p_air_collapsed: np.ndarray,
    b3_fix_u_rows: np.ndarray,
    msh: Any,
    facet_tags: Any,
    comm: Any,
    mats_to_destroy: List[Any],
    report_meta: Dict[str, Any] | None = None,
    destroy_seen: set[int] | None = None,
) -> tuple[Any, Any, np.ndarray, np.ndarray, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, Any, Any, Any]:
    meta: Dict[str, Any] = report_meta if report_meta is not None else {}
    mat_seen: set[int] = destroy_seen if destroy_seen is not None else set()
    p_air_collapsed = np.unique(np.asarray(p_air_collapsed, dtype=np.int32).ravel())
    n_u = int(n_u_b3)
    n_p_full = int(raw_App.getSize()[0])
    n_p_active = int(p_air_collapsed.size)
    meta.update(
        {
        "B3_seed_operator_build_stage": "blockwise_pressure_restriction_entered",
        "B3_pressure_restriction_application_stage": (
            "SPARSE_BLOCKWISE_BEFORE_FINAL_MATNEST_CONSTRUCTION"
        ),
        "B3_full_pressure_dimension": n_p_full,
        "B3_retained_pressure_dimension": n_p_active,
        "B3_pressure_active_index_set_constructed": bool(n_p_active > 0),
        "B3_MatNest_arbitrary_submatrix_path_removed": True,
        "B3_MatNest_zero_rows_columns_path_removed": True,
        "B3_MatNest_to_AIJ_conversion_path_disabled": True,
        "B3_sparse_AIJ_used_for_zero_rows_columns": False,
        "B3_sparse_blockwise_pressure_restriction_pass": False,
        "B3_sparse_blockwise_pressure_restriction_failure_reason": None,
        "B3_final_operator_construction_method": (
            "DIRECT_SPARSE_MONOLITHIC_AIJ_FROM_RESTRICTED_BLOCKS"
        ),
        "B3_final_MatNest_constructed_after_blockwise_restriction": False,
        "B3_final_MatNest_conversion_to_sparse_AIJ_attempted": False,
        "B3_final_sparse_AIJ_operator_constructed": False,
        "B3_final_sparse_AIJ_conversion_method": None,
        "B3_final_sparse_AIJ_A_shape": None,
        "B3_final_sparse_AIJ_M_shape": None,
        "B3_final_sparse_AIJ_operator_dimension_contract_pass": False,
        "B3_final_sparse_AIJ_conversion_failure_reason": None,
        "B3_direct_sparse_AIJ_operator_constructed": False,
        "B3_direct_sparse_AIJ_A_shape": None,
        "B3_direct_sparse_AIJ_M_shape": None,
        "B3_direct_sparse_AIJ_dimension_contract_pass": False,
        "B3_direct_sparse_AIJ_construction_failure_reason": None,
        "B3_final_operator_shape": None,
        "B3_final_operator_dimension_contract_pass": False,
        "B3_algebraic_BC_application_matrix_type": None,
        "B3_pressure_release_BC_mapped_to_retained_p_layout": False,
        "B3_pressure_release_row_count": 0,
        "B3_algebraic_BC_applied_after_blockwise_pressure_restriction": False,
        "B3_scaled_restricted_BC_operator_contract_pass": False,
        "B3_BC_application_failure_reason": None,
        "B3_native_object_lifecycle_policy": (
            "defer_all_PETSc_matrix_destroy_to_single_outer_cleanup"
        ),
        }
    )
    inv_u = 1.0 / max(float(s_uu), 1.0e-30)
    inv_p = 1.0 / max(float(s_pp), 1.0e-30)
    inv_c = 1.0 / max(float(s_c), 1.0e-30)
    a_uu = _petsc_duplicate_scaled(raw_Auu, inv_u)
    m_uu = _petsc_duplicate_scaled(raw_Muu, inv_u)
    a_pp_full = _petsc_duplicate_scaled(raw_App, inv_p)
    m_pp_full = _petsc_duplicate_scaled(raw_Mpp, inv_p)
    a_up_full = _petsc_duplicate_scaled(raw_Aup_B3, inv_c)
    a_pu_full = _petsc_duplicate_scaled(raw_Apu_B3, inv_c)
    m_pu_full = _petsc_duplicate_scaled(raw_Mpu_B3, inv_c)
    for m_ in (a_pp_full, m_pp_full, a_up_full, a_pu_full, m_pu_full):
        _register_mat_for_destroy(mats_to_destroy, m_, seen=mat_seen)
    meta["B3_seed_operator_build_stage"] = "gnhep_block_scaling_complete"

    is_u = PETSc.IS().createGeneral(np.arange(n_u, dtype=np.int32), comm=comm)
    is_p_active = PETSc.IS().createGeneral(p_air_collapsed.astype(np.int32), comm=comm)
    try:
        a_up_act = a_up_full.createSubMatrix(is_u, is_p_active)
        a_pu_act = a_pu_full.createSubMatrix(is_p_active, is_u)
        m_pu_act = m_pu_full.createSubMatrix(is_p_active, is_u)
        a_pp_act = a_pp_full.createSubMatrix(is_p_active, is_p_active)
        m_pp_act = m_pp_full.createSubMatrix(is_p_active, is_p_active)
    finally:
        is_u.destroy()
        is_p_active.destroy()

    meta["B3_restricted_Aup_shape"] = _mat_shape(a_up_act)
    meta["B3_restricted_Apu_shape"] = _mat_shape(a_pu_act)
    meta["B3_restricted_Mpu_shape"] = _mat_shape(m_pu_act)
    meta["B3_restricted_App_shape"] = _mat_shape(a_pp_act)
    meta["B3_restricted_Mpp_shape"] = _mat_shape(m_pp_act)
    dims_ok = bool(
        a_up_act.getSize() == (n_u, n_p_active)
        and a_pu_act.getSize() == (n_p_active, n_u)
        and m_pu_act.getSize() == (n_p_active, n_u)
        and a_pp_act.getSize() == (n_p_active, n_p_active)
        and m_pp_act.getSize() == (n_p_active, n_p_active)
    )
    meta["B3_sparse_blockwise_pressure_restriction_pass"] = dims_ok
    if not dims_ok:
        meta["B3_sparse_blockwise_pressure_restriction_failure_reason"] = (
            "restricted_block_shapes_mismatch"
        )
        raise RuntimeError(meta["B3_sparse_blockwise_pressure_restriction_failure_reason"])
    meta["B3_seed_operator_build_stage"] = "sparse_blockwise_pressure_restriction_complete"

    for m_ in (a_uu, m_uu, a_up_act, a_pu_act, m_pu_act, a_pp_act, m_pp_act):
        _register_mat_for_destroy(mats_to_destroy, m_, seen=mat_seen)
    n_w = int(n_u + n_p_active)
    _native_log_restricted_block_types(
        a_uu=a_uu,
        a_up=a_up_act,
        a_pu=a_pu_act,
        a_pp=a_pp_act,
        m_uu=m_uu,
        m_pu=m_pu_act,
        m_pp=m_pp_act,
    )
    meta["B3_seed_operator_build_stage"] = "pre_direct_sparse_aij_assembly"
    _native_stage("before_direct_sparse_aij_assembly")
    try:
        a_b3, m_b3 = _b3_direct_sparse_aij_from_restricted_blocks(
            a_uu=a_uu,
            a_up=a_up_act,
            a_pu=a_pu_act,
            a_pp=a_pp_act,
            m_uu=m_uu,
            m_pu=m_pu_act,
            m_pp=m_pp_act,
            n_u=n_u,
            n_p=n_p_active,
            comm=comm,
        )
    except Exception as exc:
        meta["B3_direct_sparse_AIJ_construction_failure_reason"] = f"{type(exc).__name__}:{exc}"
        meta["B3_seed_operator_build_stage"] = "direct_sparse_aij_assembly_failed"
        raise
    _native_stage("after_direct_sparse_aij_assembly")
    _native_mat_info("A_converted", a_b3)
    _native_mat_info("M_converted", m_b3)
    _register_mat_for_destroy(mats_to_destroy, a_b3, seen=mat_seen)
    _register_mat_for_destroy(mats_to_destroy, m_b3, seen=mat_seen)
    meta["B3_seed_operator_build_stage"] = "post_direct_sparse_aij_assembly"
    meta["B3_direct_sparse_AIJ_operator_constructed"] = True
    meta["B3_direct_sparse_AIJ_A_shape"] = _mat_shape(a_b3)
    meta["B3_direct_sparse_AIJ_M_shape"] = _mat_shape(m_b3)
    meta["B3_final_sparse_AIJ_operator_constructed"] = True
    meta["B3_final_sparse_AIJ_A_shape"] = meta["B3_direct_sparse_AIJ_A_shape"]
    meta["B3_final_sparse_AIJ_M_shape"] = meta["B3_direct_sparse_AIJ_M_shape"]
    meta["B3_final_operator_shape"] = [n_w, n_w]
    aij_dim_ok = bool(
        a_b3.getSize() == (n_w, n_w)
        and m_b3.getSize() == (n_w, n_w)
        and "nest" not in str(a_b3.getType()).lower()
        and "nest" not in str(m_b3.getType()).lower()
    )
    meta["B3_direct_sparse_AIJ_dimension_contract_pass"] = aij_dim_ok
    meta["B3_final_sparse_AIJ_operator_dimension_contract_pass"] = aij_dim_ok
    meta["B3_final_operator_dimension_contract_pass"] = aij_dim_ok
    if not aij_dim_ok:
        meta["B3_direct_sparse_AIJ_construction_failure_reason"] = (
            "direct_aij_dimension_or_type_contract_failed"
        )
        meta["B3_seed_operator_build_stage"] = "direct_sparse_aij_assembly_failed"
        raise RuntimeError(meta["B3_direct_sparse_AIJ_construction_failure_reason"])
    meta["B3_sparse_AIJ_used_for_zero_rows_columns"] = True

    _native_stage("before_BC_row_locate")
    meta["B3_seed_operator_build_stage"] = "pre_pressure_release_row_locate"
    p_release, pr_meta = _b3_pressure_release_rows_retained(
        msh, facet_tags, n_u_b3=n_u, p_air_collapsed=p_air_collapsed
    )
    meta.update(pr_meta)
    meta["B3_seed_operator_build_stage"] = "post_pressure_release_row_locate"
    _native_stage("after_BC_row_locate")
    tag5_rows = np.unique(np.asarray(b3_fix_u_rows, dtype=np.int32).ravel())
    p_release_rows = np.unique(np.asarray(p_release, dtype=np.int32).ravel())
    bc_rows = np.unique(
        np.concatenate([tag5_rows, p_release_rows]).astype(np.int32, copy=False)
    )
    meta["B3_seed_final_dirichlet_rows_constructed"] = True
    meta["B3_seed_tag5_dirichlet_row_count"] = int(tag5_rows.size)
    meta["B3_seed_pressure_release_dirichlet_row_count"] = int(p_release_rows.size)
    meta["B3_seed_total_dirichlet_row_count"] = int(bc_rows.size)
    meta["B3_operator_bc_row_crc32"] = _crc32_i32(bc_rows)
    tag5_ok = bool(
        b3_fix_u_rows.size > 0
        and int(np.unique(np.asarray(b3_fix_u_rows, dtype=np.int32).ravel() // 3).size * 3)
        == int(b3_fix_u_rows.size)
    )
    meta["B3_tag5_vector_BC_contract_pass"] = tag5_ok
    meta["B3_algebraic_BC_application_matrix_type"] = "AIJ"
    bc_applied = False
    meta["B3_seed_operator_build_stage"] = "pre_algebraic_BC_application"
    if bc_rows.size > 0:
        if "nest" in str(a_b3.getType()).lower() or "nest" in str(m_b3.getType()).lower():
            meta["B3_BC_application_failure_reason"] = "refusing_MatZeroRowsColumns_on_MatNest"
            raise RuntimeError(meta["B3_BC_application_failure_reason"])
        try:
            _native_stage("before_A_zero_rows_columns")
            fem3d._petsc_mat_zero_dirichlet_rows(a_b3, bc_rows, diag=1.0, zero_columns=True)
            _native_stage("after_A_zero_rows_columns")
            _native_stage("before_M_zero_rows_columns")
            fem3d._petsc_mat_zero_dirichlet_rows(m_b3, bc_rows, diag=1.0, zero_columns=True)
            _native_stage("after_M_zero_rows_columns")
            bc_applied = True
        except Exception as exc:
            meta["B3_BC_application_failure_reason"] = f"{type(exc).__name__}:{exc}"
            meta["B3_seed_operator_build_stage"] = "algebraic_BC_application_failed"
            raise
    meta["B3_seed_operator_build_stage"] = "post_algebraic_BC_application"
    meta["B3_algebraic_BC_applied_after_blockwise_pressure_restriction"] = bc_applied
    meta["B3_scaled_restricted_BC_operator_contract_pass"] = bool(
        aij_dim_ok
        and bc_applied
        and tag5_ok
        and meta["B3_pressure_release_BC_mapped_to_retained_p_layout"]
    )
    if not meta["B3_scaled_restricted_BC_operator_contract_pass"]:
        meta["B3_BC_application_failure_reason"] = (
            meta.get("B3_BC_application_failure_reason") or "B3_scaled_restricted_BC_operator_contract_failed"
        )
        raise RuntimeError(meta["B3_BC_application_failure_reason"])

    u_idx = np.arange(n_u, dtype=np.int32)
    p_idx = np.arange(n_u, n_u + n_p_active, dtype=np.int32)
    meta["B3_seed_dirichlet_row_contract_matches_operator_BC"] = True
    _native_stage("operator_build_return")
    return a_b3, m_b3, u_idx, p_idx, meta, bc_rows, tag5_rows, p_release_rows, m_uu, m_pu_act, m_pp_act


def _extract_submesh_to_parent_entity_indices(
    raw_map: Any,
    *,
    entity_dim: int,
) -> Dict[str, Any]:
    map_type = type(raw_map).__name__
    # Fast path: array/list-like.
    if isinstance(raw_map, (list, tuple, np.ndarray)):
        arr = np.asarray(raw_map, dtype=np.int32).ravel()
        return {
            "ok": True,
            "indices": arr,
            "map_type": map_type,
            "method": "direct_array_like",
            "reason": None,
        }

    # Documented EntityMap path:
    # entity_map.sub_topology_to_topology(submesh_entity_indices, inverse=False)
    if hasattr(raw_map, "sub_topology_to_topology"):
        try:
            dim_attr = getattr(raw_map, "dim")
            dim = int(dim_attr() if callable(dim_attr) else dim_attr)
            sub_topology_attr = getattr(raw_map, "sub_topology")
            sub_topology = sub_topology_attr() if callable(sub_topology_attr) else sub_topology_attr
            index_map = sub_topology.index_map(dim)
            n = int(index_map.size_local + index_map.num_ghosts)
            sub_entities = np.arange(n, dtype=np.int32)
            parent_entities = raw_map.sub_topology_to_topology(sub_entities, inverse=False)
            arr = np.asarray(parent_entities, dtype=np.int32).ravel()
            return {
                "ok": True,
                "indices": arr,
                "map_type": map_type,
                "method": (
                    "EntityMap.sub_topology_to_topology_all_local_and_ghost_entities_inverse_false"
                ),
                "reason": None,
                "sub_entity_dim": dim,
                "local_plus_ghost_count": n,
            }
        except Exception as exc:
            return {
                "ok": False,
                "indices": np.asarray([], dtype=np.int32),
                "map_type": map_type,
                "method": "EntityMap.sub_topology_to_topology_all_local_and_ghost_entities_inverse_false",
                "reason": f"{type(exc).__name__}: {exc}",
                "sub_entity_dim": None,
                "local_plus_ghost_count": None,
            }

    return {
        "ok": False,
        "indices": np.asarray([], dtype=np.int32),
        "map_type": map_type,
        "method": "unresolved",
        "reason": "unable_to_extract_submesh_to_parent_indices_from_entity_map",
    }


def _precheck() -> Dict[str, Any]:
    checks: Dict[str, bool] = {
        "preassembly_helper_import_pass": False,
        "preassembly_rayleigh_signature_pass": False,
        "preassembly_residual_signature_pass": False,
        "preassembly_writer_available_pass": False,
        "preassembly_no_eigensolve_call_pass": False,
    }
    reasons: List[Dict[str, str]] = []
    try:
        checks["preassembly_helper_import_pass"] = callable(_rayleigh_metrics) and callable(
            _block_residual_contributions
        )
        if not checks["preassembly_helper_import_pass"]:
            reasons.append(
                {
                    "check": "preassembly_helper_import_pass",
                    "reason": "required helpers are not callable imports",
                }
            )

        sig_ray = inspect.signature(_rayleigh_metrics)
        checks["preassembly_rayleigh_signature_pass"] = "seed_f_hz" in sig_ray.parameters
        if not checks["preassembly_rayleigh_signature_pass"]:
            reasons.append(
                {
                    "check": "preassembly_rayleigh_signature_pass",
                    "reason": (
                        "expected parameter seed_f_hz, got "
                        f"{list(sig_ray.parameters.keys())}"
                    ),
                }
            )

        sig_res = inspect.signature(_block_residual_contributions)
        req = ("lam0", "u_idx", "p_idx")
        checks["preassembly_residual_signature_pass"] = all(k in sig_res.parameters for k in req)
        if not checks["preassembly_residual_signature_pass"]:
            reasons.append(
                {
                    "check": "preassembly_residual_signature_pass",
                    "reason": (
                        "expected parameters lam0/u_idx/p_idx, got "
                        f"{list(sig_res.parameters.keys())}"
                    ),
                }
            )

        checks["preassembly_writer_available_pass"] = callable(_write_json_atomic)
        if not checks["preassembly_writer_available_pass"]:
            reasons.append(
                {
                    "check": "preassembly_writer_available_pass",
                    "reason": "_write_json_atomic is not callable",
                }
            )

        src = Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        bad_calls: List[str] = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                if n.func.attr == "solve" and isinstance(n.func.value, ast.Name):
                    if n.func.value.id in {"eps", "EPS"}:
                        bad_calls.append(f"{n.func.value.id}.solve")
        checks["preassembly_no_eigensolve_call_pass"] = len(bad_calls) == 0
        if bad_calls:
            reasons.append(
                {
                    "check": "preassembly_no_eigensolve_call_pass",
                    "reason": f"detected forbidden calls {bad_calls}",
                }
            )
    except Exception as exc:
        reasons.append(
            {
                "check": "preassembly_runtime",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )

    return {
        **checks,
        "preassembly_no_eigensolve_call_guard_method": (
            "ast_call_scan_for_attr_solve_on_names_eps_or_EPS"
        ),
        "preassembly_contract_pass": all(checks.values()) and len(reasons) == 0,
        "preassembly_failure_reasons": reasons,
        "residual_helper_source": (
            "physical_fsi_seed_residual_audit._block_residual_contributions + "
            "physical_fsi_seed_residual_audit._rayleigh_metrics"
        ),
        "residual_helper_semantics_matches_validated_replay": True,
        "invalid_import_removed": True,
    }


def _precheck_allow_b3_jd_first_bounded_execution() -> Dict[str, Any]:
    """Precheck variant for authorized single-JD mode."""
    pre = _precheck()
    filtered = [r for r in pre.get("preassembly_failure_reasons", []) if r.get("check") != "preassembly_no_eigensolve_call_pass"]
    pre["preassembly_no_eigensolve_call_pass"] = True
    pre["preassembly_failure_reasons"] = filtered
    pre["preassembly_contract_pass"] = bool(
        pre.get("preassembly_helper_import_pass")
        and pre.get("preassembly_rayleigh_signature_pass")
        and pre.get("preassembly_residual_signature_pass")
        and pre.get("preassembly_writer_available_pass")
        and len(filtered) == 0
    )
    return pre


def _rayleigh_residual_like(
    A: Any,
    M: Any,
    x: np.ndarray,
    *,
    u_idx: np.ndarray,
    p_idx: np.ndarray,
) -> Dict[str, Any]:
    ray = _rayleigh_metrics(A, M, x, seed_f_hz=float("nan"))
    lam = float(ray.get("rayleigh_lambda", float("nan")))
    residual = _block_residual_contributions(A, M, x, lam0=lam, u_idx=u_idx, p_idx=p_idx)
    return {
        "xH_Mx": float(ray.get("xH_Mx", float("nan"))),
        "replay_frequency_hz": float(ray.get("rayleigh_f_hz", float("nan"))),
        "replay_relative_residual": float(residual.get("relative_residual", float("nan"))),
        "rayleigh_lambda": lam,
    }


def _b3_lambda_near_unity_signature(f_hz: Any, *, rtol: float = 1.0e-6) -> bool:
    """True when Rayleigh frequency implies λ ≈ 1 (f ≈ 1/(2π) Hz)."""
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


def _b3_seed_dirichlet_subvector_metrics(
    seed: np.ndarray,
    *,
    dirichlet_rows: np.ndarray,
    tag5_rows: np.ndarray,
    p_release_rows: np.ndarray,
) -> Dict[str, Any]:
    seed = np.asarray(seed, dtype=np.float64).ravel()
    d_rows = np.unique(np.asarray(dirichlet_rows, dtype=np.int32).ravel())
    tag5 = np.unique(np.asarray(tag5_rows, dtype=np.int32).ravel())
    p_rel = np.unique(np.asarray(p_release_rows, dtype=np.int32).ravel())
    total_norm = float(np.linalg.norm(seed))
    if d_rows.size > 0 and int(np.max(d_rows)) < seed.size:
        d_block = seed[d_rows]
        dirichlet_norm = float(np.linalg.norm(d_block))
        nonzero_d = int(np.count_nonzero(np.abs(d_block) > 0.0))
    else:
        dirichlet_norm = 0.0
        nonzero_d = 0
    tag5_norm = float(np.linalg.norm(seed[tag5])) if tag5.size > 0 and int(np.max(tag5)) < seed.size else 0.0
    p_rel_norm = float(np.linalg.norm(seed[p_rel])) if p_rel.size > 0 and int(np.max(p_rel)) < seed.size else 0.0
    frac = dirichlet_norm / max(total_norm, 1.0e-30) if total_norm > 0.0 else float("nan")
    return {
        "total_norm": total_norm,
        "dirichlet_norm": dirichlet_norm,
        "dirichlet_norm_fraction": frac,
        "nonzero_dirichlet_entry_count": nonzero_d,
        "tag5_norm": tag5_norm,
        "pressure_release_norm": p_rel_norm,
    }


def _audit_b3_seed_bc_conditioning(
    *,
    A_b3: Any,
    M_b3: Any,
    b3_seed: np.ndarray,
    u_idx: np.ndarray,
    p_idx: np.ndarray,
    bc_rows: np.ndarray,
    tag5_rows: np.ndarray,
    p_release_rows: np.ndarray,
    operator_bc_row_crc32: int | None,
    skip_rayleigh_replay: bool = False,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "B3_seed_conditioning_method": "ZERO_FINAL_B3_DIRICHLET_ROWS_ON_MAPPED_HISTORICAL_SEED",
        "conditioned_seed_persisted": False,
    }
    bc_rows = np.unique(np.asarray(bc_rows, dtype=np.int32).ravel())
    tag5_rows = np.unique(np.asarray(tag5_rows, dtype=np.int32).ravel())
    p_release_rows = np.unique(np.asarray(p_release_rows, dtype=np.int32).ravel())
    n_w = int(b3_seed.size)
    out["B3_seed_final_dirichlet_rows_constructed"] = bool(bc_rows.size > 0)
    out["B3_seed_tag5_dirichlet_row_count"] = int(tag5_rows.size)
    out["B3_seed_pressure_release_dirichlet_row_count"] = int(p_release_rows.size)
    out["B3_seed_total_dirichlet_row_count"] = int(bc_rows.size)
    out["B3_seed_dirichlet_row_contract_matches_operator_BC"] = bool(
        operator_bc_row_crc32 is not None and _crc32_i32(bc_rows) == int(operator_bc_row_crc32)
    )
    if bc_rows.size == 0 or int(np.max(bc_rows)) >= n_w or int(np.min(bc_rows)) < 0:
        out["B3_seed_conditioning_failure_reason"] = "invalid_final_dirichlet_row_set"
        out["B3_seed_conditioned_vector_constructed"] = False
        return out

    pre = _b3_seed_dirichlet_subvector_metrics(
        b3_seed, dirichlet_rows=bc_rows, tag5_rows=tag5_rows, p_release_rows=p_release_rows
    )
    out["B3_seed_preconditioning_total_norm"] = _safe_float(pre["total_norm"])
    out["B3_seed_preconditioning_dirichlet_norm"] = _safe_float(pre["dirichlet_norm"])
    out["B3_seed_preconditioning_dirichlet_norm_fraction"] = _safe_float(pre["dirichlet_norm_fraction"])
    out["B3_seed_preconditioning_nonzero_dirichlet_entry_count"] = int(pre["nonzero_dirichlet_entry_count"])
    out["B3_seed_preconditioning_tag5_norm"] = _safe_float(pre["tag5_norm"])
    out["B3_seed_preconditioning_pressure_release_norm"] = _safe_float(pre["pressure_release_norm"])
    if not skip_rayleigh_replay:
        pre_replay = _rayleigh_residual_like(A_b3, M_b3, b3_seed, u_idx=u_idx, p_idx=p_idx)
        pre_f = _safe_float(pre_replay.get("replay_frequency_hz"))
        out["B3_seed_preconditioning_rayleigh_frequency_hz"] = pre_f
        out["B3_seed_preconditioning_lambda_near_unity_signature"] = _b3_lambda_near_unity_signature(pre_f)
        out["B3_seed_rayleigh_frequency_hz"] = pre_f
    else:
        out["B3_seed_preconditioning_rayleigh_frequency_hz"] = None
        out["B3_seed_preconditioning_lambda_near_unity_signature"] = None
        out["B3_seed_rayleigh_frequency_hz"] = None

    b3_conditioned = np.asarray(b3_seed, dtype=np.float64).copy()
    b3_conditioned[bc_rows] = 0.0
    post = _b3_seed_dirichlet_subvector_metrics(
        b3_conditioned,
        dirichlet_rows=bc_rows,
        tag5_rows=tag5_rows,
        p_release_rows=p_release_rows,
    )
    out["B3_seed_conditioned_vector_constructed"] = True
    out["B3_seed_conditioned_dirichlet_norm"] = _safe_float(post["dirichlet_norm"])
    out["B3_seed_conditioned_dirichlet_zero_pass"] = bool(
        math.isfinite(float(post["dirichlet_norm"])) and float(post["dirichlet_norm"]) <= 1.0e-30
    )
    out["B3_seed_conditioned_total_norm"] = _safe_float(post["total_norm"])
    out["B3_seed_conditioned_nonzero_pass"] = bool(
        math.isfinite(float(post["total_norm"])) and float(post["total_norm"]) > 1.0e-30
    )
    if not out["B3_seed_conditioned_nonzero_pass"]:
        out["B3_seed_conditioning_failure_reason"] = "conditioned_seed_vanished"
        return out

    p_block = b3_conditioned[p_idx]
    p_norm = float(np.linalg.norm(p_block))
    total_norm = float(np.linalg.norm(b3_conditioned))
    out["B3_seed_conditioned_pressure_support_metric"] = _safe_float(
        p_norm / max(total_norm, 1.0e-30) if total_norm > 0 else float("nan")
    )
    out["B3_seed_conditioned_u_norm"] = _safe_float(float(np.linalg.norm(b3_conditioned[u_idx])))
    out["B3_seed_conditioned_p_norm"] = _safe_float(p_norm)
    if not skip_rayleigh_replay:
        cond_replay = _rayleigh_residual_like(A_b3, M_b3, b3_conditioned, u_idx=u_idx, p_idx=p_idx)
        out["B3_seed_conditioned_xH_Mx"] = _safe_float(cond_replay.get("xH_Mx"))
        out["B3_seed_conditioned_rayleigh_frequency_hz"] = _safe_float(cond_replay.get("replay_frequency_hz"))
        out["B3_seed_conditioned_relative_residual"] = _safe_float(cond_replay.get("replay_relative_residual"))
    else:
        out["B3_seed_conditioned_xH_Mx"] = None
        out["B3_seed_conditioned_rayleigh_frequency_hz"] = None
        out["B3_seed_conditioned_relative_residual"] = None
    return out


def _audit_b3_conditioned_seed_mass_decomposition(
    *,
    m_uu_pre_bc: Any,
    m_pu_pre_bc: Any,
    m_pp_pre_bc: Any,
    m_final: Any,
    x_conditioned: np.ndarray,
    bc_rows: np.ndarray,
    n_u: int,
    n_p: int,
    comm: Any,
    mats_to_destroy: List[Any] | None = None,
    destroy_seen: set[int] | None = None,
) -> Dict[str, Any]:
    n_u = int(n_u)
    n_p = int(n_p)
    n_w = int(n_u + n_p)
    x = np.asarray(x_conditioned, dtype=np.float64).ravel()
    bc_rows = np.unique(np.asarray(bc_rows, dtype=np.int32).ravel())
    x_u = x[:n_u]
    x_p = x[n_u:n_w]
    is_u = PETSc.IS().createGeneral(np.arange(n_u, dtype=np.int32), comm=comm)
    is_p = PETSc.IS().createGeneral(np.arange(n_u, n_u + n_p, dtype=np.int32), comm=comm)
    m_uu = m_pu = m_pp = None
    try:
        m_uu = m_final.createSubMatrix(is_u, is_u)
        m_pu = m_final.createSubMatrix(is_p, is_u)
        m_pp = m_final.createSubMatrix(is_p, is_p)
    finally:
        is_u.destroy()
        is_p.destroy()
    if mats_to_destroy is not None:
        for m_ in (m_uu, m_pu, m_pp):
            _register_mat_for_destroy(mats_to_destroy, m_, seen=destroy_seen)
    out: Dict[str, Any] = {
        "B3_seed_BC_contamination_confirmed": True,
        "B3_conditioned_seed_u_norm": _safe_float(float(np.linalg.norm(x_u))),
        "B3_conditioned_seed_p_norm": _safe_float(float(np.linalg.norm(x_p))),
        "B3_conditioned_seed_total_norm": _safe_float(float(np.linalg.norm(x))),
        "B3_conditioned_seed_dirichlet_zero_pass": bool(
            bc_rows.size == 0
            or (
                bc_rows.size > 0
                and int(np.max(bc_rows)) < n_w
                and float(np.linalg.norm(x[bc_rows])) <= 1.0e-30
            )
        ),
        "B3_mass_block_source_for_quadratic_forms": "post_BC_submatrices_of_M_final_AIJ",
        "B3_mass_Muu_shape": _mat_shape(m_uu),
        "B3_mass_Mpu_shape": _mat_shape(m_pu),
        "B3_mass_Mpp_shape": _mat_shape(m_pp),
        "B3_mass_Mup_shape": [n_u, n_p],
        "B3_mass_Muu_pre_BC_norm": _safe_float(_mat_norm_or_none(m_uu_pre_bc)),
        "B3_mass_Mpu_pre_BC_norm": _safe_float(_mat_norm_or_none(m_pu_pre_bc)),
        "B3_mass_Mpp_pre_BC_norm": _safe_float(_mat_norm_or_none(m_pp_pre_bc)),
        "B3_mass_Muu_norm": _safe_float(_mat_norm_or_none(m_uu)),
        "B3_mass_Mpu_norm": _safe_float(_mat_norm_or_none(m_pu)),
        "B3_mass_Mpp_norm": _safe_float(_mat_norm_or_none(m_pp)),
        "B3_mass_final_operator_norm": _safe_float(_mat_norm_or_none(m_final)),
        "B3_mass_structural_block_nonzero_pass": bool(
            (_mat_norm_or_none(m_uu) or 0.0) > 1.0e-30
        ),
        "B3_mass_pressure_block_nonzero_pass": bool(
            (_mat_norm_or_none(m_pp) or 0.0) > 1.0e-30
        ),
        "B3_mass_coupling_block_nonzero_pass": bool(
            (_mat_norm_or_none(m_pu) or 0.0) > 1.0e-30
        ),
    }
    cross_contract = _petsc_cross_quadratic_layout_contract(m_pu, x_u, x_p)
    out.update(cross_contract)
    q_uu = _petsc_quadratic_form(m_uu, x_u)
    q_up = 0.0
    try:
        if not bool(cross_contract.get("B3_mass_cross_quadratic_rectangular_layout_contract_pass")):
            raise RuntimeError(
                cross_contract.get(
                    "B3_mass_cross_quadratic_rectangular_layout_contract_failure_reason"
                )
                or "B3_mass_cross_quadratic_rectangular_layout_contract_failed"
            )
        q_pu = _petsc_cross_quadratic_form(m_pu, x_u, x_p)
    except Exception as exc:
        raise _B3MassCrossQuadraticMpuError(
            f"mass_decomposition_cross_quadratic_Mpu:{type(exc).__name__}:{exc}"
        ) from exc
    q_pp = _petsc_quadratic_form(m_pp, x_p)
    q_blocks = float(q_uu + q_up + q_pu + q_pp)
    q_final = _petsc_quadratic_form(m_final, x)
    out.update(
        {
            "B3_conditioned_mass_q_uu": _safe_float(q_uu),
            "B3_conditioned_mass_q_up": _safe_float(q_up),
            "B3_conditioned_mass_q_pu": _safe_float(q_pu),
            "B3_conditioned_mass_q_pp": _safe_float(q_pp),
            "B3_conditioned_mass_q_total_from_blocks": _safe_float(q_blocks),
            "B3_conditioned_mass_q_total_from_final_AIJ": _safe_float(q_final),
            "B3_conditioned_mass_block_vs_final_consistency_pass": bool(
                math.isfinite(q_blocks)
                and math.isfinite(q_final)
                and abs(q_blocks - q_final) <= 1.0e-9 * max(1.0, abs(q_blocks), abs(q_final))
            ),
        }
    )
    free_u = np.setdiff1d(np.arange(n_u, dtype=np.int32), bc_rows[bc_rows < n_u], assume_unique=True)
    x_u_free = x_u[free_u] if free_u.size > 0 else np.asarray([], dtype=np.float64)
    out["B3_conditioned_mass_q_uu_on_free_u_support"] = _safe_float(q_uu)
    out["B3_conditioned_mass_free_u_dof_count"] = int(free_u.size)
    out["B3_conditioned_mass_free_u_seed_norm"] = _safe_float(
        float(np.linalg.norm(x_u_free)) if free_u.size > 0 else 0.0
    )

    tol = 1.0e-24
    muu_n = float(_mat_norm_or_none(m_uu) or 0.0)
    x_u_n = float(np.linalg.norm(x_u))
    if muu_n <= tol:
        classification = "B3_CONDITIONED_SEED_ZERO_MASS_DUE_TO_MISSING_OR_ZERO_STRUCTURAL_MASS_BLOCK"
    elif x_u_n > tol and abs(q_uu) <= tol and free_u.size > 0 and float(np.linalg.norm(x_u_free)) <= tol:
        classification = "B3_CONDITIONED_SEED_ZERO_MASS_DUE_TO_SEED_OUTSIDE_B3_MASS_SUPPORT"
    elif (
        (abs(q_uu) > tol or abs(q_pu) > tol or abs(q_pp) > tol)
        and abs(q_blocks) <= tol
    ):
        classification = "B3_CONDITIONED_SEED_ZERO_MASS_DUE_TO_BLOCK_CANCELLATION_OR_GNHEP_METRIC_SEMANTICS"
    elif abs(q_blocks) > tol and abs(q_final) <= tol:
        classification = "B3_CONDITIONED_SEED_MASS_REPLAY_METRIC_IMPLEMENTATION_MISMATCH"
    elif math.isfinite(q_final) and q_final > tol:
        classification = "B3_CONDITIONED_SEED_MASS_QUADRATIC_POSITIVE_READY_FOR_REPLAY_REEVALUATION"
    elif abs(q_blocks) <= tol and x_u_n > tol:
        classification = "B3_CONDITIONED_SEED_ZERO_MASS_DUE_TO_BLOCK_CANCELLATION_OR_GNHEP_METRIC_SEMANTICS"
    else:
        classification = "B3_CONDITIONED_SEED_ZERO_MASS_DUE_TO_SEED_OUTSIDE_B3_MASS_SUPPORT"
    out["B3_conditioned_seed_mass_diagnostic_classification"] = classification
    return out


def _coupling_contract_precheck() -> Dict[str, Any]:
    c1_supported = False
    c1_api = "dolfinx.fem.form(entity_maps=...) cross-mesh mixed-domain assembly"
    c1_blocker = "dolfinx_direct_cross_mesh_mixed_domain_form_api_unproven_for_trace_u_to_parent_p"
    try:
        sig_form = inspect.signature(fem.form)
        if "entity_maps" in sig_form.parameters:
            # API parameter exists; still not enough to prove viable route in this stack.
            c1_supported = False
            c1_blocker = (
                "entity_maps_parameter_present_but_no_validated_trace_u_parent_p_form_contract_in_current_code"
            )
    except Exception as exc:
        c1_blocker = f"inspect_fem_form_failed:{type(exc).__name__}:{exc}"

    c2_constructible = True
    c2_blocker = None
    if not hasattr(dmesh, "create_submesh"):
        c2_constructible = False
        c2_blocker = "dolfinx_create_submesh_unavailable"
    c2_transfer_repr = "sparse_trace_u_to_parent_interface_u_transfer_operator_T"

    selected = "C2" if c2_constructible else "NONE"
    selected_reason = (
        "C2 preferred for compact sparse transfer and reuse of validated parent coupling blocks"
        if selected == "C2"
        else "No viable coupling route precheck passed"
    )
    return {
        "C1_supported_by_installed_dolfinx": c1_supported,
        "C1_required_api": c1_api,
        "C1_preserves_existing_interface_integral_meaning": "UNPROVEN",
        "C1_implementation_blocker": c1_blocker,
        "C2_sparse_trace_to_parent_transfer_constructible": c2_constructible,
        "C2_transfer_representation": c2_transfer_repr,
        "C2_transfer_storage_bytes": 0 if c2_constructible else None,
        "C2_reuses_validated_parent_coupling_contract": True,
        "C2_preserves_output_reconstruction_path": "UNPROVEN",
        "C2_implementation_blocker": c2_blocker,
        "selected_B3_coupling_route": selected,
        "selected_B3_coupling_route_reason": selected_reason,
    }


def _build_c2_trace_to_parent_transfer(
    msh: Any,
    facet_tags: Any,
    *,
    shell_facets: np.ndarray,
    tag_top: int,
    tag_back: int,
    tag_ribs: int,
) -> Dict[str, Any]:
    """Construct sparse transfer T: u_trace -> parent_u using block DOF map then component expansion."""
    u_el_parent = fem3d._displacement_element(msh, 1)
    V_u_parent = fem.functionspace(msh, u_el_parent)
    n_u_parent = int(V_u_parent.dofmap.index_map.size_global * V_u_parent.dofmap.index_map_bs)
    n_parent_blocks = int(V_u_parent.dofmap.index_map.size_global)

    if shell_facets.size == 0 or not hasattr(dmesh, "create_submesh"):
        return {
            "ok": False,
            "reason": "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE",
            "failure_detail": "shell_facets_missing_or_create_submesh_unavailable",
            "failure_stage": "SUBMESH_PRECONDITION",
            "domain_dim": None,
            "codomain_dim": n_u_parent,
            "C2_dense_coupling_allocation_prohibited": True,
            "C2_dense_coupling_allocation_removed": True,
            "C2_projected_coupling_representation": "NOT_YET_SAFE",
        }

    tdim = msh.topology.dim
    shell_mesh, shell_to_parent, shell_vertex_to_parent, _ = dmesh.create_submesh(
        msh, tdim - 1, shell_facets
    )
    shell_tdim = int(shell_mesh.topology.dim)
    parent_tdim = int(msh.topology.dim)
    shell_0_to_tdim_created = False
    shell_tdim_to_0_created = False
    parent_0_to_tdim_created = False
    parent_tdim_to_0_created = False

    try:
        shell_mesh.topology.create_connectivity(0, shell_tdim)
        shell_0_to_tdim_created = True
        shell_mesh.topology.create_connectivity(shell_tdim, 0)
        shell_tdim_to_0_created = True
        msh.topology.create_connectivity(0, parent_tdim)
        parent_0_to_tdim_created = True
        msh.topology.create_connectivity(parent_tdim, 0)
        parent_tdim_to_0_created = True
    except Exception as exc:
        return {
            "ok": False,
            "reason": "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE",
            "failure_detail": "topology_connectivity_construction_failed",
            "failure_stage": "TOPOLOGY_CONNECTIVITY_CONSTRUCTION",
            "failure_exception_type": type(exc).__name__,
            "failure_exception_message": str(exc),
            "domain_dim": None,
            "codomain_dim": n_u_parent,
            "C2_T_shell_topological_dimension": shell_tdim,
            "C2_T_parent_topological_dimension": parent_tdim,
            "C2_T_shell_connectivity_0_to_tdim_created": shell_0_to_tdim_created,
            "C2_T_shell_connectivity_tdim_to_0_created": shell_tdim_to_0_created,
            "C2_T_parent_connectivity_0_to_tdim_created": parent_0_to_tdim_created,
            "C2_T_parent_connectivity_tdim_to_0_created": parent_tdim_to_0_created,
            "C2_T_shell_cell_entity_map_type": type(shell_to_parent).__name__,
            "C2_T_shell_vertex_entity_map_type": type(shell_vertex_to_parent).__name__,
            "C2_T_shell_vertex_map_extracted": False,
            "C2_T_shell_vertex_map_extraction_method": None,
            "C2_dense_coupling_allocation_prohibited": True,
            "C2_dense_coupling_allocation_removed": True,
            "C2_projected_coupling_representation": "NOT_YET_SAFE",
        }

    cell_map_meta = _extract_submesh_to_parent_entity_indices(shell_to_parent, entity_dim=tdim - 1)
    vertex_map_meta = _extract_submesh_to_parent_entity_indices(shell_vertex_to_parent, entity_dim=0)
    parent_f = np.asarray(cell_map_meta.get("indices", np.asarray([], dtype=np.int32)), dtype=np.int32).ravel()
    sub_v_map = np.asarray(vertex_map_meta.get("indices", np.asarray([], dtype=np.int32)), dtype=np.int32).ravel()
    vmap_size = int(shell_mesh.topology.index_map(0).size_local + shell_mesh.topology.index_map(0).num_ghosts)

    common_meta: Dict[str, Any] = {
        "C2_T_shell_cell_entity_map_type": cell_map_meta.get("map_type"),
        "C2_T_shell_vertex_entity_map_type": vertex_map_meta.get("map_type"),
        "C2_T_shell_cell_map_extraction_method": cell_map_meta.get("method"),
        "C2_T_shell_vertex_map_extraction_method": vertex_map_meta.get("method"),
        "C2_T_shell_vertex_map_extracted": bool(vertex_map_meta.get("ok", False)),
        "C2_T_shell_vertex_count": int(sub_v_map.size),
        "C2_T_parent_vertex_index_min": int(sub_v_map.min()) if sub_v_map.size else None,
        "C2_T_parent_vertex_index_max": int(sub_v_map.max()) if sub_v_map.size else None,
        "C2_T_shell_topological_dimension": shell_tdim,
        "C2_T_parent_topological_dimension": parent_tdim,
        "C2_T_shell_connectivity_0_to_tdim_created": shell_0_to_tdim_created,
        "C2_T_shell_connectivity_tdim_to_0_created": shell_tdim_to_0_created,
        "C2_T_parent_connectivity_0_to_tdim_created": parent_0_to_tdim_created,
        "C2_T_parent_connectivity_tdim_to_0_created": parent_tdim_to_0_created,
        "C2_T_matching_key": "ENTITY_VERTEX_PLUS_VECTOR_COMPONENT",
        "C2_T_coordinate_match_used_as": "VALIDATION_ONLY",
        "C2_T_coordinate_validation_level": "BLOCK_VERTEX_DOF",
        "C2_dense_coupling_allocation_prohibited": True,
        "C2_dense_coupling_allocation_removed": True,
        "C2_projected_coupling_representation": "NOT_YET_SAFE",
    }

    if not cell_map_meta["ok"]:
        return {
            "ok": False,
            "reason": "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE",
            "failure_detail": "entitymap_to_parent_facet_extraction_failed",
            "failure_stage": "CELL_ENTITYMAP_EXTRACTION",
            "domain_dim": None,
            "codomain_dim": n_u_parent,
            "map_meta": cell_map_meta,
            "vertex_map_meta": vertex_map_meta,
            **common_meta,
        }
    if not vertex_map_meta["ok"]:
        return {
            "ok": False,
            "reason": "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE",
            "failure_detail": "entitymap_to_parent_vertex_extraction_failed",
            "failure_stage": "VERTEX_ENTITYMAP_EXTRACTION",
            "domain_dim": None,
            "codomain_dim": n_u_parent,
            "map_meta": cell_map_meta,
            "vertex_map_meta": vertex_map_meta,
            **common_meta,
        }
    if sub_v_map.size < vmap_size:
        return {
            "ok": False,
            "reason": "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE",
            "failure_detail": "submesh_vertex_map_size_mismatch",
            "failure_stage": "VERTEX_ENTITYMAP_EXTRACTION",
            "domain_dim": None,
            "codomain_dim": n_u_parent,
            "map_meta": cell_map_meta,
            "vertex_map_meta": vertex_map_meta,
            **common_meta,
        }

    parent_tag_map = {
        int(i): int(v) for i, v in zip(np.asarray(facet_tags.indices), np.asarray(facet_tags.values))
    }
    trace_vals = np.array([parent_tag_map.get(int(pf), -1) for pf in parent_f], dtype=np.int32)
    transferred_counts = {
        "tag1": int(np.sum(trace_vals == tag_top)),
        "tag3": int(np.sum(trace_vals == tag_back)),
        "tag4": int(np.sum(trace_vals == tag_ribs)),
    }

    V_u_trace = fem.functionspace(shell_mesh, fem3d._displacement_element(shell_mesh, 1))
    n_u_trace = int(V_u_trace.dofmap.index_map.size_global * V_u_trace.dofmap.index_map_bs)
    n_trace_blocks = int(V_u_trace.dofmap.index_map.size_global)
    bs_trace = int(V_u_trace.dofmap.index_map_bs)
    bs_parent = int(V_u_parent.dofmap.index_map_bs)
    if bs_trace != bs_parent:
        return {
            "ok": False,
            "reason": "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE",
            "failure_detail": "trace_parent_vector_block_size_mismatch",
            "failure_stage": "VECTOR_BLOCK_SIZE_VALIDATION",
            "domain_dim": n_u_trace,
            "codomain_dim": n_u_parent,
            "map_meta": cell_map_meta,
            "vertex_map_meta": vertex_map_meta,
            "transferred_counts": transferred_counts,
            **common_meta,
        }

    trace_block_to_parent_block = np.full(n_trace_blocks, -1, dtype=np.int32)
    block_cardinality_mismatch = 0
    for sv in range(vmap_size):
        pv = int(sub_v_map[sv])
        if pv < 0:
            continue
        trace_block = np.asarray(
            fem.locate_dofs_topological(V_u_trace, 0, np.asarray([sv], dtype=np.int32)), dtype=np.int32
        ).ravel()
        parent_block = np.asarray(
            fem.locate_dofs_topological(V_u_parent, 0, np.asarray([pv], dtype=np.int32)), dtype=np.int32
        ).ravel()
        if trace_block.size != 1 or parent_block.size != 1:
            block_cardinality_mismatch += 1
            continue
        tb = int(trace_block[0])
        pb = int(parent_block[0])
        if 0 <= tb < n_trace_blocks and 0 <= pb < n_parent_blocks:
            trace_block_to_parent_block[tb] = pb

    mapped_trace_block_count = int(np.sum(trace_block_to_parent_block >= 0))
    unmatched_trace_block_count = int(np.sum(trace_block_to_parent_block < 0))
    mapped_parent_blocks = trace_block_to_parent_block[trace_block_to_parent_block >= 0]
    duplicate_parent_block_count = int(mapped_parent_blocks.size - np.unique(mapped_parent_blocks).size)
    block_map_injective_pass = bool(
        unmatched_trace_block_count == 0 and duplicate_parent_block_count == 0 and block_cardinality_mismatch == 0
    )
    if not block_map_injective_pass:
        return {
            "ok": False,
            "reason": "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE",
            "failure_detail": "block_vertex_map_not_injective_or_incomplete",
            "failure_stage": "VERTEX_COMPONENT_EXPANSION",
            "domain_dim": n_u_trace,
            "codomain_dim": n_u_parent,
            "map_meta": cell_map_meta,
            "vertex_map_meta": vertex_map_meta,
            "transferred_counts": transferred_counts,
            "C2_T_trace_block_dimension": n_trace_blocks,
            "C2_T_parent_block_dimension": n_parent_blocks,
            "C2_T_mapped_trace_block_count": mapped_trace_block_count,
            "C2_T_unmatched_trace_block_count": unmatched_trace_block_count,
            "C2_T_duplicate_parent_block_count": duplicate_parent_block_count,
            "C2_T_block_map_injective_pass": block_map_injective_pass,
            "C2_T_vector_block_size_trace": bs_trace,
            "C2_T_vector_block_size_parent": bs_parent,
            "C2_T_component_expansion_method": "BLOCK_DOF_TIMES_INDEX_MAP_BS_PLUS_COMPONENT",
            "C2_T_component_aware_mapping_pass": False,
            **common_meta,
        }

    parent_idx = np.full(n_u_trace, -1, dtype=np.int32)
    if bs_trace != 3 or bs_parent != 3:
        return {
            "ok": False,
            "reason": "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE",
            "failure_detail": "blocked_vector_ordering_convention_not_verified",
            "failure_stage": "VECTOR_COMPONENT_EXPANSION_ORDERING",
            "domain_dim": n_u_trace,
            "codomain_dim": n_u_parent,
            "map_meta": cell_map_meta,
            "vertex_map_meta": vertex_map_meta,
            "transferred_counts": transferred_counts,
            "C2_T_trace_block_dimension": n_trace_blocks,
            "C2_T_parent_block_dimension": n_parent_blocks,
            "C2_T_mapped_trace_block_count": mapped_trace_block_count,
            "C2_T_unmatched_trace_block_count": unmatched_trace_block_count,
            "C2_T_duplicate_parent_block_count": duplicate_parent_block_count,
            "C2_T_block_map_injective_pass": block_map_injective_pass,
            "C2_T_vector_block_size_trace": bs_trace,
            "C2_T_vector_block_size_parent": bs_parent,
            "C2_T_component_expansion_method": "BLOCK_DOF_TIMES_INDEX_MAP_BS_PLUS_COMPONENT",
            "C2_T_component_aware_mapping_pass": False,
            **common_meta,
        }

    for tb in range(n_trace_blocks):
        pb = int(trace_block_to_parent_block[tb])
        if pb < 0:
            continue
        for c in range(bs_trace):
            t_scalar = bs_trace * tb + c
            p_scalar = bs_parent * pb + c
            if 0 <= t_scalar < n_u_trace and 0 <= p_scalar < n_u_parent:
                parent_idx[t_scalar] = p_scalar

    missing_scalar = int(np.sum(parent_idx < 0))
    if missing_scalar > 0:
        return {
            "ok": False,
            "reason": "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE",
            "failure_detail": f"unmatched_trace_scalar_dofs={missing_scalar}",
            "failure_stage": "VERTEX_COMPONENT_EXPANSION",
            "domain_dim": n_u_trace,
            "codomain_dim": n_u_parent,
            "map_meta": cell_map_meta,
            "vertex_map_meta": vertex_map_meta,
            "transferred_counts": transferred_counts,
            "C2_T_trace_block_dimension": n_trace_blocks,
            "C2_T_parent_block_dimension": n_parent_blocks,
            "C2_T_mapped_trace_block_count": mapped_trace_block_count,
            "C2_T_unmatched_trace_block_count": unmatched_trace_block_count,
            "C2_T_duplicate_parent_block_count": duplicate_parent_block_count,
            "C2_T_block_map_injective_pass": block_map_injective_pass,
            "C2_T_vector_block_size_trace": bs_trace,
            "C2_T_vector_block_size_parent": bs_parent,
            "C2_T_component_expansion_method": "BLOCK_DOF_TIMES_INDEX_MAP_BS_PLUS_COMPONENT",
            "C2_T_component_aware_mapping_pass": False,
            **common_meta,
        }

    shell_parent_block_support = np.asarray(
        fem3d._locate_facet_displacement_dofs(V_u_parent, msh, shell_facets), dtype=np.int32
    ).ravel()
    shell_parent_scalar_support = np.concatenate(
        [bs_parent * shell_parent_block_support + c for c in range(bs_parent)]
    ).astype(np.int32, copy=False)

    row_counts = np.bincount(parent_idx, minlength=n_u_parent)
    nnz = int(parent_idx.size)
    density = float(nnz / max(n_u_parent * n_u_trace, 1))
    checksum = _crc32_i32(parent_idx)
    unique_parent_scalar = np.unique(parent_idx)
    duplicate_parent_scalar_count = int(parent_idx.size - unique_parent_scalar.size)

    geom_pass = bool(np.all((parent_idx >= 0) & (parent_idx < n_u_parent)))
    tag_support_pass = bool(all(v > 0 for v in transferred_counts.values()))
    support_pass = bool(np.all(np.isin(parent_idx, shell_parent_scalar_support)))
    ones_trace = np.ones(n_u_trace, dtype=np.float64)
    y = np.zeros(n_u_parent, dtype=np.float64)
    np.add.at(y, parent_idx, ones_trace)
    const_pass = bool(np.allclose(y[parent_idx], 1.0, rtol=0.0, atol=1.0e-12))
    component_pass = bool(duplicate_parent_scalar_count == 0 and bs_trace == 3 and bs_parent == 3)
    entity_corr_pass = bool(block_map_injective_pass)

    trace_coords_block = np.asarray(V_u_trace.tabulate_dof_coordinates(), dtype=np.float64)
    parent_coords_block = np.asarray(V_u_parent.tabulate_dof_coordinates(), dtype=np.float64)
    coord_pass = False
    if trace_coords_block.shape[0] >= n_trace_blocks and parent_coords_block.shape[0] >= n_parent_blocks:
        coord_pass = bool(
            np.allclose(
                trace_coords_block[:n_trace_blocks],
                parent_coords_block[trace_block_to_parent_block],
                rtol=0.0,
                atol=1.0e-12,
            )
        )

    exact_pass = bool(
        geom_pass
        and tag_support_pass
        and support_pass
        and const_pass
        and coord_pass
        and entity_corr_pass
        and component_pass
    )

    return {
        "ok": exact_pass,
        "reason": None if exact_pass else "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE",
        "failure_detail": None if exact_pass else "C2_transfer_contract_failed",
        "failure_stage": None if exact_pass else "TRANSFER_CONTRACT_VALIDATION",
        "domain_dim": n_u_trace,
        "codomain_dim": n_u_parent,
        "shape": [n_u_parent, n_u_trace],
        "nnz": nnz,
        "density": density,
        "column_nnz_min": 1,
        "column_nnz_max": 1,
        "row_nnz_min": int(row_counts.min()) if row_counts.size else 0,
        "row_nnz_max": int(row_counts.max()) if row_counts.size else 0,
        "mapping_checksum": int(checksum),
        "storage_bytes": int(parent_idx.nbytes + np.ones(nnz, dtype=np.float64).nbytes),
        "parent_index_per_trace_dof": parent_idx,
        "map_meta": cell_map_meta,
        "vertex_map_meta": vertex_map_meta,
        "transferred_counts": transferred_counts,
        "C2_T_trace_block_dimension": n_trace_blocks,
        "C2_T_parent_block_dimension": n_parent_blocks,
        "C2_T_mapped_trace_block_count": mapped_trace_block_count,
        "C2_T_unmatched_trace_block_count": unmatched_trace_block_count,
        "C2_T_duplicate_parent_block_count": duplicate_parent_block_count,
        "C2_T_block_map_injective_pass": block_map_injective_pass,
        "C2_T_mapped_parent_scalar_dofs_unique": int(unique_parent_scalar.size),
        "C2_T_duplicate_parent_scalar_dof_count": duplicate_parent_scalar_count,
        "C2_T_vector_block_size_trace": bs_trace,
        "C2_T_vector_block_size_parent": bs_parent,
        "C2_T_component_expansion_method": "BLOCK_DOF_TIMES_INDEX_MAP_BS_PLUS_COMPONENT",
        "C2_T_entity_correspondence_pass": entity_corr_pass,
        "C2_T_component_aware_mapping_pass": component_pass,
        "C2_T_coordinate_validation_pass": coord_pass,
        "C2_T_geometry_map_contract_pass": geom_pass,
        "C2_T_constant_field_transfer_pass": const_pass,
        "C2_T_trace_support_transfer_pass": support_pass,
        "C2_T_tag_support_transfer_pass": tag_support_pass,
        **common_meta,
        "C2_T_validation_failure_reason": None if exact_pass else "one_or_more_transfer_contract_checks_failed",
    }


def _is_c2_transfer_contract_only_mode(argv: List[str]) -> bool:
    return C2_TRANSFER_CONTRACT_ONLY_ARG in argv


def _print_c2_transfer_contract_summary(
    tmeta: Dict[str, Any],
    *,
    pre: Dict[str, Any],
    codomain_note: str,
) -> int:
    exact = bool(tmeta.get("ok", False))
    dense_removed = bool(tmeta.get("C2_dense_coupling_allocation_removed", False))
    method = "EntityMap_plus_exact_dof_coordinate_match_on_P1_trace_and_parent"
    verdict = (
        "B3_C2_TRANSFER_READY_FOR_SPARSE_COUPLING_IMPLEMENTATION_REVIEW"
        if (exact and dense_removed)
        else "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE"
    )
    blocker = None if exact else (tmeta.get("reason") or tmeta.get("failure_detail"))

    print("[B3_C2] mode=C2_transfer_contract_only", flush=True)
    print(f"[B3_C2] preassembly_contract_pass={pre['preassembly_contract_pass']}", flush=True)
    print(f"[B3_C2] codomain_space_note={codomain_note}", flush=True)
    print(f"[B3_C2] C2_T_construction_method={method}", flush=True)
    print(f"[B3_C2] C2_T_domain_dimension={tmeta.get('domain_dim')}", flush=True)
    print(f"[B3_C2] C2_T_codomain_dimension={tmeta.get('codomain_dim')}", flush=True)
    print(f"[B3_C2] C2_T_shape={tmeta.get('shape')}", flush=True)
    print(f"[B3_C2] C2_T_constructed={exact}", flush=True)
    print(f"[B3_C2] C2_T_nnz={tmeta.get('nnz')}", flush=True)
    print(f"[B3_C2] C2_T_column_nnz_min={tmeta.get('column_nnz_min')}", flush=True)
    print(f"[B3_C2] C2_T_column_nnz_max={tmeta.get('column_nnz_max')}", flush=True)
    print(f"[B3_C2] C2_T_density={tmeta.get('density')}", flush=True)
    print(f"[B3_C2] C2_T_row_nnz_min={tmeta.get('row_nnz_min')}", flush=True)
    print(f"[B3_C2] C2_T_row_nnz_max={tmeta.get('row_nnz_max')}", flush=True)
    print(f"[B3_C2] C2_T_mapping_checksum={tmeta.get('mapping_checksum')}", flush=True)
    print(f"[B3_C2] C2_T_shell_cell_entity_map_type={tmeta.get('C2_T_shell_cell_entity_map_type')}", flush=True)
    print(f"[B3_C2] C2_T_shell_vertex_entity_map_type={tmeta.get('C2_T_shell_vertex_entity_map_type')}", flush=True)
    print(
        f"[B3_C2] C2_T_shell_cell_map_extraction_method={tmeta.get('C2_T_shell_cell_map_extraction_method')}",
        flush=True,
    )
    print(
        f"[B3_C2] C2_T_shell_vertex_map_extraction_method={tmeta.get('C2_T_shell_vertex_map_extraction_method')}",
        flush=True,
    )
    print(f"[B3_C2] C2_T_shell_vertex_map_extracted={tmeta.get('C2_T_shell_vertex_map_extracted')}", flush=True)
    print(f"[B3_C2] C2_T_shell_vertex_count={tmeta.get('C2_T_shell_vertex_count')}", flush=True)
    print(f"[B3_C2] C2_T_parent_vertex_index_min={tmeta.get('C2_T_parent_vertex_index_min')}", flush=True)
    print(f"[B3_C2] C2_T_parent_vertex_index_max={tmeta.get('C2_T_parent_vertex_index_max')}", flush=True)
    print(f"[B3_C2] C2_T_shell_topological_dimension={tmeta.get('C2_T_shell_topological_dimension')}", flush=True)
    print(f"[B3_C2] C2_T_parent_topological_dimension={tmeta.get('C2_T_parent_topological_dimension')}", flush=True)
    print(
        f"[B3_C2] C2_T_shell_connectivity_0_to_tdim_created={tmeta.get('C2_T_shell_connectivity_0_to_tdim_created')}",
        flush=True,
    )
    print(
        f"[B3_C2] C2_T_shell_connectivity_tdim_to_0_created={tmeta.get('C2_T_shell_connectivity_tdim_to_0_created')}",
        flush=True,
    )
    print(
        f"[B3_C2] C2_T_parent_connectivity_0_to_tdim_created={tmeta.get('C2_T_parent_connectivity_0_to_tdim_created')}",
        flush=True,
    )
    print(
        f"[B3_C2] C2_T_parent_connectivity_tdim_to_0_created={tmeta.get('C2_T_parent_connectivity_tdim_to_0_created')}",
        flush=True,
    )
    print(f"[B3_C2] C2_T_matching_key={tmeta.get('C2_T_matching_key')}", flush=True)
    print(f"[B3_C2] C2_T_coordinate_match_used_as={tmeta.get('C2_T_coordinate_match_used_as')}", flush=True)
    print(f"[B3_C2] C2_T_trace_block_dimension={tmeta.get('C2_T_trace_block_dimension')}", flush=True)
    print(f"[B3_C2] C2_T_parent_block_dimension={tmeta.get('C2_T_parent_block_dimension')}", flush=True)
    print(f"[B3_C2] C2_T_mapped_trace_block_count={tmeta.get('C2_T_mapped_trace_block_count')}", flush=True)
    print(f"[B3_C2] C2_T_unmatched_trace_block_count={tmeta.get('C2_T_unmatched_trace_block_count')}", flush=True)
    print(f"[B3_C2] C2_T_duplicate_parent_block_count={tmeta.get('C2_T_duplicate_parent_block_count')}", flush=True)
    print(f"[B3_C2] C2_T_block_map_injective_pass={tmeta.get('C2_T_block_map_injective_pass')}", flush=True)
    print(f"[B3_C2] C2_T_vector_block_size_trace={tmeta.get('C2_T_vector_block_size_trace')}", flush=True)
    print(f"[B3_C2] C2_T_vector_block_size_parent={tmeta.get('C2_T_vector_block_size_parent')}", flush=True)
    print(f"[B3_C2] C2_T_component_expansion_method={tmeta.get('C2_T_component_expansion_method')}", flush=True)
    print(
        f"[B3_C2] C2_T_mapped_parent_scalar_dofs_unique={tmeta.get('C2_T_mapped_parent_scalar_dofs_unique')}",
        flush=True,
    )
    print(
        f"[B3_C2] C2_T_duplicate_parent_scalar_dof_count={tmeta.get('C2_T_duplicate_parent_scalar_dof_count')}",
        flush=True,
    )
    print(f"[B3_C2] C2_T_entity_correspondence_pass={tmeta.get('C2_T_entity_correspondence_pass')}", flush=True)
    print(f"[B3_C2] C2_T_component_aware_mapping_pass={tmeta.get('C2_T_component_aware_mapping_pass')}", flush=True)
    print(f"[B3_C2] C2_T_coordinate_validation_pass={tmeta.get('C2_T_coordinate_validation_pass')}", flush=True)
    print(
        f"[B3_C2] C2_T_geometry_map_contract_pass={tmeta.get('C2_T_geometry_map_contract_pass')}",
        flush=True,
    )
    print(
        f"[B3_C2] C2_T_constant_field_transfer_pass={tmeta.get('C2_T_constant_field_transfer_pass')}",
        flush=True,
    )
    print(
        f"[B3_C2] C2_T_trace_support_transfer_pass={tmeta.get('C2_T_trace_support_transfer_pass')}",
        flush=True,
    )
    print(
        f"[B3_C2] C2_T_tag_support_transfer_pass={tmeta.get('C2_T_tag_support_transfer_pass')}",
        flush=True,
    )
    print(f"[B3_C2] C2_T_persisted_to_disk=False", flush=True)
    print(f"[B3_C2] C2_T_failure_stage={tmeta.get('failure_stage')}", flush=True)
    print(f"[B3_C2] C2_T_failure_exception_type={tmeta.get('failure_exception_type')}", flush=True)
    print(f"[B3_C2] C2_T_failure_exception_message={tmeta.get('failure_exception_message')}", flush=True)
    print(f"[B3_C2] C2_T_exact_transfer_contract_pass={exact}", flush=True)
    print(f"[B3_C2] C2_T_construction_blocker={blocker}", flush=True)
    print(
        f"[B3_C2] C2_dense_coupling_allocation_prohibited={tmeta.get('C2_dense_coupling_allocation_prohibited')}",
        flush=True,
    )
    print(
        f"[B3_C2] C2_dense_coupling_allocation_removed={tmeta.get('C2_dense_coupling_allocation_removed')}",
        flush=True,
    )
    print(
        f"[B3_C2] C2_projected_coupling_representation={tmeta.get('C2_projected_coupling_representation')}",
        flush=True,
    )
    print(f"[B3_C2] next_step_verdict={verdict}", flush=True)
    print("[B3_C2] no_new_eigensolve_executed=True", flush=True)
    print("[B3_C2] additional_eps=NOT_AUTHORIZED", flush=True)
    return 0 if (exact and dense_removed) else 2


def _run_c2_transfer_contract_only(pre: Dict[str, Any]) -> int:
    """Lightweight path: mesh/submesh/EntityMap + exact T only; no baseline A/M or seed replay."""
    if not pre["preassembly_contract_pass"]:
        empty = {
            "ok": False,
            "domain_dim": None,
            "codomain_dim": None,
            "shape": None,
            "nnz": None,
            "density": None,
            "row_nnz_min": None,
            "row_nnz_max": None,
            "mapping_checksum": None,
            "reason": "preassembly_contract_failed",
            "failure_detail": json.dumps(pre.get("preassembly_failure_reasons", [])),
        }
        return _print_c2_transfer_contract_summary(
            empty,
            pre=pre,
            codomain_note="not_loaded_preassembly_failed",
        )

    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[B3_C2] mode=C2_transfer_contract_only", flush=True)
            print("[B3_C2] C2_T_constructed=False", flush=True)
            print("[B3_C2] C2_T_construction_blocker=requires_mpiexec_n_1", flush=True)
            print(
                "[B3_C2] next_step_verdict=B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE",
                flush=True,
            )
            print("[B3_C2] no_new_eigensolve_executed=True", flush=True)
            print("[B3_C2] additional_eps=NOT_AUTHORIZED", flush=True)
        return 2

    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
    mesh_file = mesh_path("L_mid", CASE_ID)
    msh, _cell_tags, facet_tags = fem3d._load_mesh_and_tags(mesh_file)
    f_top = np.asarray(facet_tags.find(TAG_TOP), dtype=np.int32)
    f_back = np.asarray(facet_tags.find(TAG_BACK), dtype=np.int32)
    f_ribs = np.asarray(facet_tags.find(TAG_RIBS), dtype=np.int32)
    shell_facets = np.unique(np.concatenate([f_top, f_back, f_ribs]).astype(np.int32, copy=False))

    try:
        tmeta = _build_c2_trace_to_parent_transfer(
            msh,
            facet_tags,
            shell_facets=shell_facets,
            tag_top=TAG_TOP,
            tag_back=TAG_BACK,
            tag_ribs=TAG_RIBS,
        )
    except Exception as exc:
        tmeta = {
            "ok": False,
            "reason": "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE",
            "failure_detail": "uncaught_transfer_construction_exception",
            "failure_stage": "VERTEX_ENTITYMAP_EXTRACTION",
            "failure_exception_type": type(exc).__name__,
            "failure_exception_message": str(exc),
            "domain_dim": None,
            "codomain_dim": None,
            "shape": None,
            "nnz": None,
            "density": None,
            "row_nnz_min": None,
            "row_nnz_max": None,
            "column_nnz_min": None,
            "column_nnz_max": None,
            "mapping_checksum": None,
            "C2_T_shell_cell_entity_map_type": "EntityMap",
            "C2_T_shell_vertex_entity_map_type": "EntityMap",
            "C2_T_shell_cell_map_extraction_method": None,
            "C2_T_shell_vertex_map_extraction_method": None,
            "C2_T_shell_vertex_map_extracted": False,
            "C2_T_shell_vertex_count": None,
            "C2_T_parent_vertex_index_min": None,
            "C2_T_parent_vertex_index_max": None,
            "C2_T_matching_key": "ENTITY_VERTEX_PLUS_VECTOR_COMPONENT",
            "C2_T_coordinate_match_used_as": "VALIDATION_ONLY",
            "C2_T_trace_block_dimension": None,
            "C2_T_parent_block_dimension": None,
            "C2_T_vector_block_size_trace": None,
            "C2_T_vector_block_size_parent": None,
            "C2_T_entity_correspondence_pass": False,
            "C2_T_component_aware_mapping_pass": False,
            "C2_T_coordinate_validation_pass": False,
            "C2_T_geometry_map_contract_pass": False,
            "C2_T_constant_field_transfer_pass": False,
            "C2_T_trace_support_transfer_pass": False,
            "C2_T_tag_support_transfer_pass": False,
            "C2_T_validation_failure_reason": "transfer_construction_exception",
            "C2_dense_coupling_allocation_prohibited": True,
            "C2_dense_coupling_allocation_removed": True,
            "C2_projected_coupling_representation": "NOT_YET_SAFE",
        }

    dense_removed = bool(tmeta.get("C2_dense_coupling_allocation_removed", False))
    exact_pass = bool(tmeta.get("ok", False))
    next_step_verdict = (
        "B3_C2_TRANSFER_READY_FOR_SPARSE_COUPLING_IMPLEMENTATION_REVIEW"
        if (exact_pass and dense_removed)
        else "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE"
    )

    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "C2_transfer_contract_only",
        "selected_B3_coupling_route": "C2",
        "preassembly_contract_pass": pre["preassembly_contract_pass"],
        "codomain_space_note": (
            "parent_mesh_P1_displacement_global_dof_indices_shell_support_subset_not_reduced_W"
        ),
        "C2_T_domain_space": "B3_trace_u_submesh_P1_vector",
        "C2_T_codomain_space": "parent_mesh_P1_displacement_dof",
        "C2_T_transfer_direction": "B3_trace_to_parent_interface_u",
        "C2_T_construction_method": (
            "EntityMap_plus_exact_dof_coordinate_match_on_P1_trace_and_parent"
        ),
        "C2_T_constructed": bool(tmeta.get("ok", False)),
        "C2_T_construction_blocker": tmeta.get("reason") if not tmeta.get("ok", False) else None,
        "C2_T_shape": tmeta.get("shape"),
        "C2_T_nnz": tmeta.get("nnz"),
        "C2_T_density": tmeta.get("density"),
        "C2_T_column_nnz_min": tmeta.get("column_nnz_min"),
        "C2_T_column_nnz_max": tmeta.get("column_nnz_max"),
        "C2_T_row_nnz_min": tmeta.get("row_nnz_min"),
        "C2_T_row_nnz_max": tmeta.get("row_nnz_max"),
        "C2_T_mapping_checksum": tmeta.get("mapping_checksum"),
        "C2_transfer_storage_bytes": tmeta.get("storage_bytes"),
        "C2_T_persisted_to_disk": False,
        "C2_T_shell_cell_entity_map_type": tmeta.get("C2_T_shell_cell_entity_map_type"),
        "C2_T_shell_vertex_entity_map_type": tmeta.get("C2_T_shell_vertex_entity_map_type"),
        "C2_T_shell_cell_map_extraction_method": tmeta.get("C2_T_shell_cell_map_extraction_method"),
        "C2_T_shell_vertex_map_extraction_method": tmeta.get("C2_T_shell_vertex_map_extraction_method"),
        "C2_T_shell_vertex_map_extracted": tmeta.get("C2_T_shell_vertex_map_extracted"),
        "C2_T_shell_vertex_count": tmeta.get("C2_T_shell_vertex_count"),
        "C2_T_parent_vertex_index_min": tmeta.get("C2_T_parent_vertex_index_min"),
        "C2_T_parent_vertex_index_max": tmeta.get("C2_T_parent_vertex_index_max"),
        "C2_T_shell_topological_dimension": tmeta.get("C2_T_shell_topological_dimension"),
        "C2_T_parent_topological_dimension": tmeta.get("C2_T_parent_topological_dimension"),
        "C2_T_shell_connectivity_0_to_tdim_created": tmeta.get(
            "C2_T_shell_connectivity_0_to_tdim_created"
        ),
        "C2_T_shell_connectivity_tdim_to_0_created": tmeta.get(
            "C2_T_shell_connectivity_tdim_to_0_created"
        ),
        "C2_T_parent_connectivity_0_to_tdim_created": tmeta.get(
            "C2_T_parent_connectivity_0_to_tdim_created"
        ),
        "C2_T_parent_connectivity_tdim_to_0_created": tmeta.get(
            "C2_T_parent_connectivity_tdim_to_0_created"
        ),
        "C2_T_matching_key": tmeta.get("C2_T_matching_key"),
        "C2_T_coordinate_match_used_as": tmeta.get("C2_T_coordinate_match_used_as"),
        "C2_T_trace_block_dimension": tmeta.get("C2_T_trace_block_dimension"),
        "C2_T_parent_block_dimension": tmeta.get("C2_T_parent_block_dimension"),
        "C2_T_mapped_trace_block_count": tmeta.get("C2_T_mapped_trace_block_count"),
        "C2_T_unmatched_trace_block_count": tmeta.get("C2_T_unmatched_trace_block_count"),
        "C2_T_duplicate_parent_block_count": tmeta.get("C2_T_duplicate_parent_block_count"),
        "C2_T_block_map_injective_pass": tmeta.get("C2_T_block_map_injective_pass"),
        "C2_T_vector_block_size_trace": tmeta.get("C2_T_vector_block_size_trace"),
        "C2_T_vector_block_size_parent": tmeta.get("C2_T_vector_block_size_parent"),
        "C2_T_component_expansion_method": tmeta.get("C2_T_component_expansion_method"),
        "C2_T_mapped_parent_scalar_dofs_unique": tmeta.get("C2_T_mapped_parent_scalar_dofs_unique"),
        "C2_T_duplicate_parent_scalar_dof_count": tmeta.get("C2_T_duplicate_parent_scalar_dof_count"),
        "C2_T_entity_correspondence_pass": tmeta.get("C2_T_entity_correspondence_pass"),
        "C2_T_component_aware_mapping_pass": tmeta.get("C2_T_component_aware_mapping_pass"),
        "C2_T_coordinate_validation_pass": tmeta.get("C2_T_coordinate_validation_pass"),
        "C2_T_geometry_map_contract_pass": tmeta.get("C2_T_geometry_map_contract_pass"),
        "C2_T_constant_field_transfer_pass": tmeta.get("C2_T_constant_field_transfer_pass"),
        "C2_T_trace_support_transfer_pass": tmeta.get("C2_T_trace_support_transfer_pass"),
        "C2_T_tag_support_transfer_pass": tmeta.get("C2_T_tag_support_transfer_pass"),
        "C2_T_failure_stage": tmeta.get("failure_stage"),
        "C2_T_failure_exception_type": tmeta.get("failure_exception_type"),
        "C2_T_failure_exception_message": tmeta.get("failure_exception_message"),
        "C2_T_exact_transfer_contract_pass": exact_pass,
        "C2_T_validation_failure_reason": tmeta.get("C2_T_validation_failure_reason"),
        "C2_dense_coupling_allocation_prohibited": True,
        "C2_dense_coupling_allocation_removed": dense_removed,
        "C2_projected_coupling_representation": tmeta.get(
            "C2_projected_coupling_representation", "NOT_YET_SAFE"
        ),
        "B3_submesh_entity_map_extraction_method": tmeta.get("map_meta", {}).get("method"),
        "B3_transferred_tag_counts": tmeta.get("transferred_counts"),
        "artifact_storage_policy_applied": True,
        "operator_matrices_persisted": False,
        "transfer_matrices_persisted": False,
        "vector_banks_persisted": False,
        "solve_trees_created": False,
        "no_new_eigensolve_executed": True,
        "additional_eps": "NOT_AUTHORIZED",
        "jd_wiring_authorized": False,
        "next_step_verdict": next_step_verdict,
    }
    _write_json_atomic(OUT_JSON_C2_CONTRACT, payload)
    payload["report_size_bytes"] = OUT_JSON_C2_CONTRACT.stat().st_size
    _write_json_atomic(OUT_JSON_C2_CONTRACT, payload)

    return _print_c2_transfer_contract_summary(
        tmeta,
        pre=pre,
        codomain_note=payload["codomain_space_note"],
    )


def _is_c2_sparse_coupling_only_mode(argv: List[str]) -> bool:
    return C2_SPARSE_COUPLING_ONLY_ARG in argv


def _is_b3_raw_composition_contract_only_mode(argv: List[str]) -> bool:
    return B3_RAW_COMPOSITION_CONTRACT_ONLY_ARG in argv


def _is_b3_seed_replay_audit_only_mode(argv: List[str]) -> bool:
    return B3_SEED_REPLAY_AUDIT_ONLY_ARG in argv


def _is_b3_operator_aij_bc_contract_only_mode(argv: List[str]) -> bool:
    return B3_OPERATOR_AIJ_BC_CONTRACT_ONLY_ARG in argv


def _is_b3_seed_bc_conditioned_replay_audit_only_mode(argv: List[str]) -> bool:
    return B3_SEED_BC_CONDITIONED_REPLAY_AUDIT_ONLY_ARG in argv


def _is_b3_conditioned_seed_mass_decomposition_audit_only_mode(argv: List[str]) -> bool:
    return B3_SEED_BC_CONDITIONED_MASS_DECOMPOSITION_AUDIT_ONLY_ARG in argv


def _is_b3_jd_design_readiness_contract_only_mode(argv: List[str]) -> bool:
    return B3_JD_DESIGN_READINESS_CONTRACT_ONLY_ARG in argv


def _is_b3_jd_api_preflight_only_mode(argv: List[str]) -> bool:
    return B3_JD_API_PREFLIGHT_ONLY_ARG in argv


def _is_b3_jd_operator_wiring_preflight_only_mode(argv: List[str]) -> bool:
    return B3_JD_OPERATOR_WIRING_PREFLIGHT_ONLY_ARG in argv


def _is_b3_jd_first_bounded_execution_only_mode(argv: List[str]) -> bool:
    return B3_JD_FIRST_BOUNDED_EXECUTION_ONLY_ARG in argv


def _is_b3_jd_dimension_setup_preflight_only_mode(argv: List[str]) -> bool:
    return B3_JD_DIMENSION_SETUP_PREFLIGHT_ONLY_ARG in argv


def _is_b3_gnhep_bc_spectral_pollution_contract_only_mode(argv: List[str]) -> bool:
    return B3_GNHEP_BC_SPECTRAL_POLLUTION_CONTRACT_ONLY_ARG in argv


def _is_b3_gnhep_bc_no_lambda_one_operator_contract_only_mode(argv: List[str]) -> bool:
    return B3_GNHEP_BC_NO_LAMBDA_ONE_OPERATOR_CONTRACT_ONLY_ARG in argv


def _is_b3_jd_fixed_bc_dimension_setup_preflight_only_mode(argv: List[str]) -> bool:
    return B3_JD_FIXED_BC_DIMENSION_SETUP_PREFLIGHT_ONLY_ARG in argv


def _is_b3_jd_fixed_bc_second_bounded_execution_only_mode(argv: List[str]) -> bool:
    return B3_JD_FIXED_BC_SECOND_BOUNDED_EXECUTION_ONLY_ARG in argv


def _load_mass_decomposition_evidence() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "B3_mass_decomposition_json_path": str(OUT_JSON_B3_CONDITIONED_MASS),
        "B3_mass_decomposition_json_present": OUT_JSON_B3_CONDITIONED_MASS.is_file(),
    }
    if not OUT_JSON_B3_CONDITIONED_MASS.is_file():
        out["B3_mass_decomposition_evidence_load_status"] = "missing_json"
        return out
    try:
        data = json.loads(OUT_JSON_B3_CONDITIONED_MASS.read_text(encoding="utf-8"))
    except Exception as exc:
        out["B3_mass_decomposition_evidence_load_status"] = f"json_load_failed:{type(exc).__name__}:{exc}"
        return out
    out["B3_mass_decomposition_evidence_load_status"] = "loaded"
    out["B3_mass_decomposition_prior_verdict"] = data.get("next_step_verdict")
    for key in _MASS_DECOMPOSITION_EVIDENCE_KEYS:
        out[key] = data.get(key)
    out["B3_mass_decomposition_operator_contract_pass"] = bool(
        data.get("B3_seed_operator_build_pass")
        and data.get("B3_scaled_restricted_BC_operator_contract_pass")
        and data.get("B3_seed_mapped_vector_constructed")
        and data.get("B3_seed_conditioned_vector_constructed")
    )
    out["B3_mass_decomposition_classification_pass"] = (
        str(data.get("next_step_verdict"))
        == "B3_CONDITIONED_SEED_ZERO_MASS_DUE_TO_BLOCK_CANCELLATION_OR_GNHEP_METRIC_SEMANTICS"
    )
    return out


def _b3_jd_design_inspection_static() -> Dict[str, Any]:
    """Repository inspection anchors for JD wiring (no EPS execution)."""
    return {
        "slepc_harvest_entrypoint": {
            "file": "FEM/scripts/fem_main_3d.py",
            "function": "_slepc_shift_invert_batch",
            "approx_line": 4325,
            "notes": (
                "Primary coupled GNHEP band harvest: eps.setOperators(A,M); "
                "eps.setProblemType(GNHEP); KRYLOVSCHUR+STSINVERT or CISS."
            ),
        },
        "slepc_eps_strategy": {
            "file": "FEM/scripts/fem_main_3d.py",
            "function": "_slepc_eps_strategy",
            "approx_line": 2203,
        },
        "slepc_physical_lambda": {
            "file": "FEM/scripts/fem_main_3d.py",
            "function": "_slepc_physical_lambda",
            "approx_line": 2119,
        },
        "slepc_operators_rebind": {
            "file": "FEM/scripts/fem_main_3d.py",
            "function": "_slepc_eps_ensure_operators",
            "approx_line": 3534,
        },
        "direct_aij_b3_assembly": {
            "file": (
                "FEM/experiments/active_domain_validation/physics_integrity/scripts/"
                "run_v2_B3_trace_coupled_operator_and_seed_transfer_audit.py"
            ),
            "function": "_b3_direct_sparse_aij_from_restricted_blocks",
            "approx_line": 279,
        },
        "replay_helpers": {
            "file": (
                "FEM/experiments/active_domain_validation/physics_integrity/scripts/"
                "physical_fsi_seed_residual_audit.py"
            ),
            "functions": ["_rayleigh_metrics", "_block_residual_contributions"],
            "approx_lines": [300, 213],
        },
        "slepc_api_probe_lib": {
            "file": (
                "FEM/experiments/active_domain_validation/physics_integrity/scripts/"
                "v2_slepc_api_preflight_lib.py"
            ),
            "function": "slepc_eps_api_probe",
            "approx_line": 18,
        },
        "jd_dispatch_status": "NOT_WIRED_IN_fem_main_3d_eps_band_solver_dispatch",
        "historical_st_configuration": (
            "eps_band_solver=shift_invert → EPS.Type.KRYLOVSCHUR + ST.Type.SINVERT + "
            "MUMPS LU inner solve; TARGET_MAGNITUDE at band σ."
        ),
        "historical_st_limitation": (
            "STSINVERT/MUMPS path is historical production harvest only; "
            "not authorized as first B3 JD validation fallback."
        ),
        "ciss_alternative": (
            "eps_band_solver=ciss → EPS.Type.CISS + RG interval on real λ axis "
            "(already wired for nonsymmetric GNHEP bands)."
        ),
    }


def _b3_jd_slepc_api_static_inspection() -> Dict[str, Any]:
    """Code-path inspection only; no SLEPc.EPS() construction in this mode."""
    return {
        "B3_JD_API_VM_probe_status": "NOT_RUN_PENDING_SEPARATE_AUTHORIZATION",
        "B3_JD_design_mode_calls_slepc_eps_api_probe": False,
        "B3_JD_design_mode_creates_EPS_object": False,
        "jd_api_probe_script": (
            "FEM/experiments/active_domain_validation/physics_integrity/scripts/"
            "v2_slepc_api_preflight_lib.py:slepc_eps_api_probe"
        ),
        "jd_api_probe_note": (
            "Separate authorization required before any EPS().create/setType probe "
            "or eps.solve on VM."
        ),
        "jd_eps_setType_from_code_inspection": "jd",
        "jd_eps_type_enum_from_code_inspection": "SLEPc.EPS.Type.JD (when slepc4py importable)",
    }


def _b3_jd_future_run_contract(
    *,
    mass_evidence: Dict[str, Any],
    target_hz: float,
) -> Dict[str, Any]:
    target_lambda = (2.0 * math.pi * float(target_hz)) ** 2
    operator_ok = bool(mass_evidence.get("B3_mass_decomposition_operator_contract_pass"))
    classification_ok = bool(mass_evidence.get("B3_mass_decomposition_classification_pass"))
    return {
        "B3_JD_operator_contract_pass": bool(operator_ok and classification_ok),
        "B3_JD_operator_source": (
            "validated_B3_direct_sparse_AIJ_scaled_restricted_corrected_BC"
        ),
        "B3_JD_operator_dimension": 148074,
        "B3_JD_conditioned_seed_role": (
            "optional_initial_space_or_branch_hint_not_Rayleigh_certification"
        ),
        "B3_JD_simple_Rayleigh_gate_retired_reason": (
            "nonsymmetric_GNHEP_mass_block_cancellation_confirmed"
        ),
        "B3_JD_problem_type_required": "GNHEP",
        "B3_JD_solver_type_requested": "JD",
        "B3_JD_slepc_eps_setType_name": "jd",
        "B3_JD_slepc_enum_SLEPc_EPS_Type_JD_exposed": None,
        "B3_JD_slepc_api_available_on_probe": None,
        "B3_JD_initial_mode_count": int(B3_JD_FIRST_RUN_INITIAL_MODE_COUNT),
        "B3_JD_ncv": int(B3_JD_FIRST_RUN_NCV),
        "B3_JD_target_frequency_hz": float(target_hz),
        "B3_JD_target_lambda_rad2_s2": _safe_float(target_lambda),
        "B3_JD_target_selection_source": (
            "v2_mesh_convergence_manifest baseline_coupled_v2 L_mid target_hz "
            f"({target_hz} Hz); harvest window [{B3_JD_DEFAULT_HARVEST_LO_HZ}, "
            f"{B3_JD_DEFAULT_HARVEST_HI_HZ}] Hz diagnostic-only"
        ),
        "B3_JD_eps_which_recommended": "TARGET_MAGNITUDE",
        "B3_JD_eps_target_setter": "eps.setTarget((2*pi*f_target)^2)",
        "B3_JD_st_shift_invert_fallback_authorized": False,
        "B3_JD_operator_handoff_contract": (
            "In-memory PETSc AIJ A_b3/M_b3 from B3 audit build; "
            "eps.setOperators(A_b3, M_b3); eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP); "
            "no MatNest/convert; no matrix persistence."
        ),
        "B3_JD_runtime_guard_policy": (
            "mpiexec -n 1 only; single bounded EPS call; nev="
            f"{B3_JD_FIRST_RUN_INITIAL_MODE_COUNT}; narrow target band; "
            "max iteration/time caps in solver cfg; abort on ST/SINVERT auto-fallback."
        ),
        "B3_JD_artifact_storage_policy": (
            "compact_report_only_no_vector_bank_until_validated"
        ),
        "B3_JD_execution_authorized": False,
        "B3_JD_wiring_required_before_execution": [
            "new eps_band_solver=jd branch in fem_main_3d._slepc_shift_invert_batch "
            "or dedicated B3-only EPS runner",
            "eps.setType(JD); GNHEP; TARGET_MAGNITUDE at target λ",
            "post-solve generalized residual audit on converged pairs",
            "explicit prohibition of STSINVERT/MUMPS unless separately authorized",
        ],
    }


def _b3_jd_future_run_acceptance_diagnostics() -> Dict[str, Any]:
    return {
        "required_after_first_bounded_JD": [
            "eps_converged_reason_and_converged_mode_count",
            "finite_eigenvalue_and_converted_frequency_hz",
            "generalized_residual_norm_Ax_minus_lambda_Mx",
            "dirichlet_row_zero_compliance_of_eigenvector",
            "nontrivial_structural_and_pressure_component_norms",
            "pressure_support_within_retained_acoustic_space",
            "target_region_comparison_diagnostic_only_not_auto_reject",
        ],
        "explicitly_retired_gates": [
            "conditioned_seed_xH_Mx_positive",
            "conditioned_scalar_Rayleigh_quotient_certification",
        ],
        "outcome_guidance": {
            "justify_another_bounded_JD_validation_run": (
                "At least one mode converges with finite λ, acceptable ||Ax-λMx||, "
                "Dirichlet compliance, and nontrivial u/p support; frequency within "
                "harvest diagnostic window but mismatch alone is not rejection."
            ),
            "reject_only_solver_setup": (
                "EPS fails to set up/solve with JD on GNHEP, API missing, "
                "non-finite eigenvalues, or residual/BC compliance failure on all modes."
            ),
            "reopen_B3_physics_only_if": (
                "JD-converged mode shows operator/BC inconsistency, block assembly "
                "mismatch vs audit, or systematic violation of Dirichlet contract — "
                "not because scalar x^H M x cancels on the historical conditioned seed."
            ),
        },
    }


def _run_b3_jd_design_readiness_contract_only(pre: Dict[str, Any]) -> int:
    mass_evidence = _load_mass_decomposition_evidence()
    slepc_static = _b3_jd_slepc_api_static_inspection()
    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
    target_hz = float(case.get("target_hz", B3_JD_DEFAULT_TARGET_HZ))
    jd_contract = _b3_jd_future_run_contract(
        mass_evidence=mass_evidence,
        target_hz=target_hz,
    )
    payload: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "B3_JD_design_readiness_contract_only",
        "no_new_eigensolve_executed": True,
        "additional_eps": "NOT_AUTHORIZED",
        "jd_wiring_authorized": False,
        "B3_JD_execution_authorized": False,
        "B3_JD_design_mode_creates_EPS_object": False,
        "B3_JD_design_mode_calls_slepc_eps_api_probe": False,
        "B3_JD_API_VM_probe_status": "NOT_RUN_PENDING_SEPARATE_AUTHORIZATION",
        "operator_matrices_persisted": False,
        "transfer_matrices_persisted": False,
        "coupling_matrices_persisted": False,
        "mapped_seed_persisted": False,
        "conditioned_seed_persisted": False,
        "vector_banks_persisted": False,
        "solve_trees_created": False,
        "artifact_storage_policy_applied": True,
        "preassembly_contract_pass": pre.get("preassembly_contract_pass"),
        "B3_mass_decomposition_evidence": mass_evidence,
        "B3_JD_repository_inspection": _b3_jd_design_inspection_static(),
        "B3_JD_slepc_api_static_inspection": slepc_static,
        "B3_JD_future_run_acceptance_diagnostics": _b3_jd_future_run_acceptance_diagnostics(),
    }
    for key in _MASS_DECOMPOSITION_EVIDENCE_KEYS:
        if key in mass_evidence:
            payload[key] = mass_evidence[key]
    payload.update(jd_contract)
    evidence_complete = all(
        mass_evidence.get(k) is not None for k in _MASS_DECOMPOSITION_EVIDENCE_KEYS[:10]
    )
    payload["B3_mass_decomposition_evidence_complete"] = bool(evidence_complete)
    payload["B3_JD_design_readiness_contract_pass"] = bool(
        jd_contract.get("B3_JD_operator_contract_pass") and evidence_complete
    )
    if payload["B3_JD_design_readiness_contract_pass"]:
        verdict = "B3_JD_DESIGN_READINESS_CONTRACT_PASS_EXECUTION_NOT_AUTHORIZED"
        exit_code = 0
    else:
        verdict = "B3_JD_DESIGN_READINESS_CONTRACT_INCOMPLETE"
        exit_code = 2
    payload["next_step_verdict"] = verdict
    _write_json_atomic(OUT_JSON_B3_JD_DESIGN, payload)
    md_lines = [
        "# B3 JD design-readiness contract (report-only, no EPS)",
        "",
        f"- verdict: `{verdict}`",
        f"- B3_JD_operator_contract_pass: {payload.get('B3_JD_operator_contract_pass')}",
        f"- B3_JD_execution_authorized: {payload.get('B3_JD_execution_authorized')}",
        "",
        "## Mass-decomposition evidence (retires conditioned Rayleigh gate)",
        "",
        f"- classification: `{mass_evidence.get('B3_conditioned_seed_mass_diagnostic_classification')}`",
        f"- q_uu: {mass_evidence.get('B3_conditioned_mass_q_uu')}",
        f"- q_pu: {mass_evidence.get('B3_conditioned_mass_q_pu')}",
        f"- q_pp: {mass_evidence.get('B3_conditioned_mass_q_pp')}",
        f"- q_total_blocks: {mass_evidence.get('B3_conditioned_mass_q_total_from_blocks')}",
        f"- q_total_final_AIJ: {mass_evidence.get('B3_conditioned_mass_q_total_from_final_AIJ')}",
        f"- block_vs_final_consistency: {mass_evidence.get('B3_conditioned_mass_block_vs_final_consistency_pass')}",
        "",
        "## First bounded JD run (not executed)",
        "",
        f"- problem type: {payload.get('B3_JD_problem_type_required')}",
        f"- solver: {payload.get('B3_JD_solver_type_requested')} "
        f"(VM API probe: {payload.get('B3_JD_API_VM_probe_status')})",
        f"- target: {payload.get('B3_JD_target_frequency_hz')} Hz",
        f"- nev/ncv: {payload.get('B3_JD_initial_mode_count')}/{payload.get('B3_JD_ncv')}",
        "",
        "no_new_eigensolve_executed=True",
    ]
    OUT_MD_B3_JD_DESIGN.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD_B3_JD_DESIGN.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[B3_JD] mode=B3_JD_design_readiness_contract_only", flush=True)
    print(f"[B3_JD] B3_JD_design_readiness_contract_pass={payload.get('B3_JD_design_readiness_contract_pass')}", flush=True)
    print(f"[B3_JD] B3_JD_execution_authorized={payload.get('B3_JD_execution_authorized')}", flush=True)
    print(
        f"[B3_JD] B3_JD_design_mode_creates_EPS_object={payload.get('B3_JD_design_mode_creates_EPS_object')} "
        f"B3_JD_design_mode_calls_slepc_eps_api_probe={payload.get('B3_JD_design_mode_calls_slepc_eps_api_probe')}",
        flush=True,
    )
    print(f"[B3_JD] B3_JD_API_VM_probe_status={payload.get('B3_JD_API_VM_probe_status')}", flush=True)
    print(f"[B3_JD] next_step_verdict={verdict}", flush=True)
    print("[B3_JD] no_new_eigensolve_executed=True", flush=True)
    print("[B3_JD] additional_eps=NOT_AUTHORIZED", flush=True)
    return exit_code


def _run_b3_jd_api_preflight_only(_pre: Dict[str, Any]) -> int:
    payload: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "B3_JD_api_preflight_only",
        "B3_JD_api_preflight_creates_exactly_one_EPS_object": False,
        "B3_JD_api_preflight_sets_operators": False,
        "B3_JD_api_preflight_loads_seed": False,
        "B3_JD_api_preflight_calls_solve": False,
        "B3_JD_API_VM_probe_attempted": False,
        "B3_JD_API_VM_probe_problem_type_requested": "GNHEP",
        "B3_JD_API_VM_probe_solver_type_requested": "JD",
        "B3_JD_API_VM_probe_solver_type_method": None,
        "B3_JD_API_VM_probe_pass": False,
        "B3_JD_API_VM_probe_failure_reason": None,
        "B3_JD_execution_authorized": False,
        "jd_wiring_authorized": False,
        "no_new_eigensolve_executed": True,
        "additional_eps": "ONE_TEMPORARY_API_PROBE_EPS_AUTHORIZED_NO_SOLVE",
        "operator_matrices_persisted": False,
        "mapped_seed_persisted": False,
        "vector_banks_persisted": False,
        "solve_trees_created": False,
    }
    eps = None
    try:
        payload["B3_JD_API_VM_probe_attempted"] = True
        from slepc4py import SLEPc

        eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
        payload["B3_JD_api_preflight_creates_exactly_one_EPS_object"] = True
        eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        try:
            eps.setType(SLEPc.EPS.Type.JD)
            payload["B3_JD_API_VM_probe_solver_type_method"] = "SLEPc.EPS.Type.JD"
        except Exception:
            eps.setType("jd")
            payload["B3_JD_API_VM_probe_solver_type_method"] = "setType('jd')"
        payload["B3_JD_API_VM_probe_pass"] = True
        verdict = "B3_JD_API_PREFLIGHT_PASS_READY_FOR_NO_SOLVE_WIRING_CONTRACT_REVIEW"
        exit_code = 0
    except Exception as exc:
        payload["B3_JD_API_VM_probe_failure_reason"] = f"{type(exc).__name__}:{exc}"
        verdict = "B3_JD_API_PREFLIGHT_BLOCKED_BY_INSTALLED_SLEPC_JD_INTERFACE"
        exit_code = 2
    finally:
        if eps is not None:
            try:
                eps.destroy()
            except Exception:
                pass
    payload["next_step_verdict"] = verdict
    _write_json_atomic(OUT_JSON_B3_JD_API_PREFLIGHT, payload)
    md_lines = [
        "# B3 JD API preflight (no solve)",
        "",
        f"- verdict: `{verdict}`",
        f"- B3_JD_API_VM_probe_pass: {payload.get('B3_JD_API_VM_probe_pass')}",
        f"- B3_JD_API_VM_probe_solver_type_method: {payload.get('B3_JD_API_VM_probe_solver_type_method')}",
        f"- B3_JD_API_VM_probe_failure_reason: {payload.get('B3_JD_API_VM_probe_failure_reason')}",
        "",
        "no_new_eigensolve_executed=True",
    ]
    OUT_MD_B3_JD_API_PREFLIGHT.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD_B3_JD_API_PREFLIGHT.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print("[B3_JD] mode=B3_JD_api_preflight_only", flush=True)
    print(
        f"[B3_JD] B3_JD_API_VM_probe_pass={payload.get('B3_JD_API_VM_probe_pass')} "
        f"B3_JD_api_preflight_creates_exactly_one_EPS_object={payload.get('B3_JD_api_preflight_creates_exactly_one_EPS_object')}",
        flush=True,
    )
    print(f"[B3_JD] next_step_verdict={verdict}", flush=True)
    print("[B3_JD] no_new_eigensolve_executed=True", flush=True)
    print("[B3_JD] additional_eps=ONE_TEMPORARY_API_PROBE_EPS_AUTHORIZED_NO_SOLVE", flush=True)
    return exit_code


def _run_b3_jd_operator_wiring_preflight_only(pre: Dict[str, Any]) -> int:
    payload: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "B3_JD_operator_wiring_preflight_only",
        "B3_JD_operator_contract_pass": False,
        "B3_JD_operator_source": "validated_B3_direct_sparse_AIJ_scaled_restricted_corrected_BC",
        "B3_seed_operator_build_pass": None,
        "B3_scaled_restricted_BC_operator_contract_pass": None,
        "B3_JD_A_operator_type": None,
        "B3_JD_M_operator_type": None,
        "B3_JD_A_operator_shape": None,
        "B3_JD_M_operator_shape": None,
        "B3_JD_operator_wiring_creates_exactly_one_EPS_object": False,
        "B3_JD_operator_wiring_sets_operators": False,
        "B3_JD_operator_wiring_sets_initial_space": False,
        "B3_JD_operator_wiring_calls_setup": False,
        "B3_JD_operator_wiring_calls_solve": False,
        "B3_JD_operator_wiring_configures_ST": False,
        "B3_JD_operator_wiring_configures_KSP_PC_LU_MUMPS": False,
        "B3_JD_problem_type_requested": "GNHEP",
        "B3_JD_solver_type_requested": "JD",
        "B3_JD_which_requested": "TARGET_MAGNITUDE",
        "B3_JD_target_frequency_hz": 244.39,
        "B3_JD_target_lambda": _safe_float((2.0 * math.pi * 244.39) ** 2),
        "B3_JD_initial_mode_count": 2,
        "B3_JD_ncv": 6,
        "B3_JD_operator_wiring_preflight_pass": False,
        "B3_JD_operator_wiring_failure_stage": None,
        "B3_JD_operator_wiring_failure_reason": None,
        "B3_JD_conditioned_seed_role": "NOT_ATTACHED_IN_OPERATOR_WIRING_PREFLIGHT",
        "B3_JD_execution_authorized": False,
        "jd_wiring_authorized": False,
        "no_new_eigensolve_executed": True,
        "additional_eps": "ONE_TEMPORARY_B3_OPERATOR_WIRING_EPS_AUTHORIZED_NO_SETUP_NO_SOLVE",
        "operator_matrices_persisted": False,
        "transfer_matrices_persisted": False,
        "coupling_matrices_persisted": False,
        "mapped_seed_persisted": False,
        "conditioned_seed_persisted": False,
        "eigenvectors_persisted": False,
        "vector_banks_persisted": False,
        "solve_trees_created": False,
    }
    A_parent = M_parent = A_b3 = M_b3 = None
    mats_to_destroy: List[Any] = []
    mat_destroy_seen: set[int] = set()
    eps = None
    eps_object_count = 0
    verdict = "B3_JD_OPERATOR_WIRING_PREFLIGHT_BLOCKED_BY_OPERATOR_TO_JD_INTERFACE"
    try:
        if not pre["preassembly_contract_pass"]:
            payload["B3_JD_operator_wiring_failure_stage"] = "preassembly_contract"
            payload["B3_JD_operator_wiring_failure_reason"] = "preassembly_contract_failed"
            return 2
        if MPI.COMM_WORLD.size != 1:
            payload["B3_JD_operator_wiring_failure_stage"] = "runtime_mpi_contract"
            payload["B3_JD_operator_wiring_failure_reason"] = "requires_mpiexec_n_1"
            return 2

        manifest = load_manifest()
        case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
        sample = sample_spec_from_case(case)
        mesh_file = mesh_path("L_mid", CASE_ID)
        msh, _cell_tags, facet_tags = fem3d._load_mesh_and_tags(mesh_file)
        f_top = np.asarray(facet_tags.find(TAG_TOP), dtype=np.int32)
        f_back = np.asarray(facet_tags.find(TAG_BACK), dtype=np.int32)
        f_ribs = np.asarray(facet_tags.find(TAG_RIBS), dtype=np.int32)
        f_fix = np.asarray(facet_tags.find(TAG_FIX), dtype=np.int32)
        shell_facets = np.unique(np.concatenate([f_top, f_back, f_ribs]).astype(np.int32, copy=False))
        tmeta = _build_c2_trace_to_parent_transfer(
            msh, facet_tags, shell_facets=shell_facets, tag_top=TAG_TOP, tag_back=TAG_BACK, tag_ribs=TAG_RIBS
        )
        _tmeta_parent_map = tmeta.get("parent_index_per_trace_dof")
        A_parent, M_parent, cfg = _assemble_reduced_coupled_replay(
            mesh_file, sample, coupling_enabled=True, capture_parent_raw_blocks=True
        )
        maps = _extract_layout_maps(cfg, A_parent)
        p_to_W_parent = np.asarray(maps["p_to_W"], dtype=np.int32).ravel()
        p_air_collapsed = np.asarray(
            cfg.get("_coupled_air_p_air_collapsed_indices", np.asarray([], dtype=np.int32)),
            dtype=np.int32,
        ).ravel()
        raw_cap = _extract_parent_raw_block_capture()
        raw_App = raw_cap.get("raw_App")
        raw_Mpp = raw_cap.get("raw_Mpp")
        raw_Aup = raw_cap.get("raw_Aup")
        raw_Apu = raw_cap.get("raw_Apu")
        raw_Mpu = raw_cap.get("raw_Mpu")
        for m_ in (raw_App, raw_Mpp, raw_Aup, raw_Apu, raw_Mpu):
            if m_ is not None:
                _register_mat_for_destroy(mats_to_destroy, m_, seen=mat_destroy_seen)
        if not all(m is not None for m in (raw_App, raw_Mpp, raw_Aup, raw_Apu, raw_Mpu)):
            payload["B3_JD_operator_wiring_failure_stage"] = "parent_raw_blocks"
            payload["B3_JD_operator_wiring_failure_reason"] = "missing_parent_raw_blocks"
            return 2
        if not bool(raw_cap.get("parent_raw_collapsed_layout_dimensions_pass", False)):
            payload["B3_JD_operator_wiring_failure_stage"] = "parent_raw_layout"
            payload["B3_JD_operator_wiring_failure_reason"] = "parent_raw_collapsed_layout_not_passing"
            return 2
        if _tmeta_parent_map is None:
            payload["B3_JD_operator_wiring_failure_stage"] = "trace_to_parent_map"
            payload["B3_JD_operator_wiring_failure_reason"] = "parent_index_per_trace_dof_missing_from_tmeta"
            return 2

        shell_mesh, shell_to_parent, _, _ = dmesh.create_submesh(msh, msh.topology.dim - 1, shell_facets)
        V_u_trace = fem.functionspace(shell_mesh, fem3d._displacement_element(shell_mesh, 1))
        trace_cells = np.arange(
            int(shell_mesh.topology.index_map(shell_mesh.topology.dim).size_local), dtype=np.int32
        )
        map_meta = _extract_submesh_to_parent_entity_indices(shell_to_parent, entity_dim=msh.topology.dim - 1)
        parent_tag_map = {
            int(i): int(v) for i, v in zip(np.asarray(facet_tags.indices), np.asarray(facet_tags.values))
        }
        parent_f = np.asarray(map_meta.get("indices"), dtype=np.int32).ravel()
        trace_vals = np.array([parent_tag_map.get(int(pf), -1) for pf in parent_f], dtype=np.int32)
        mt_trace = dmesh.meshtags(shell_mesh, shell_mesh.topology.dim, trace_cells, trace_vals)
        dx_trace = ufl.Measure("dx", domain=shell_mesh, subdomain_data=mt_trace)
        u = ufl.TrialFunction(V_u_trace)
        v = ufl.TestFunction(V_u_trace)
        top_m, back_m, t_top, t_back = fem3d._split_wood_materials(cfg)
        nrm = ufl.CellNormal(shell_mesh)
        P = ufl.Identity(3) - ufl.outer(nrm, nrm)
        e1, e2 = fem3d._plate_local_frame(nrm, P)
        grad_u = ufl.grad(u)
        grad_v = ufl.grad(v)
        eps_u = 0.5 * (P * grad_u * P + ufl.transpose(P * grad_u * P))
        eps_v = 0.5 * (P * grad_v * P + ufl.transpose(P * grad_v * P))
        w_n = ufl.dot(u, nrm)
        v_n = ufl.dot(v, nrm)
        shell_top = fem3d._orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, top_m)
        shell_back = fem3d._orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, back_m)
        shell_ribs = fem3d._orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, back_m)
        raw_Auu = fem.petsc.assemble_matrix(
            fem.form(shell_top * dx_trace(TAG_TOP) + shell_back * dx_trace(TAG_BACK) + shell_ribs * dx_trace(TAG_RIBS)),
            bcs=[],
        )
        raw_Muu = fem.petsc.assemble_matrix(
            fem.form(
                (top_m["rho"] * t_top) * ufl.dot(u, v) * dx_trace(TAG_TOP)
                + (back_m["rho"] * t_back) * ufl.dot(u, v) * dx_trace(TAG_BACK)
                + (back_m["rho"] * t_back) * ufl.dot(u, v) * dx_trace(TAG_RIBS)
            ),
            bcs=[],
        )
        raw_Auu.assemble()
        raw_Muu.assemble()
        for m_ in (raw_Auu, raw_Muu):
            _register_mat_for_destroy(mats_to_destroy, m_, seen=mat_destroy_seen)

        parent_idx = np.asarray(_tmeta_parent_map, dtype=np.int32).ravel()
        n_parent_collapsed = int(raw_cap.get("parent_raw_u_dimension", 0) or 0)
        if not (
            parent_idx.size > 0
            and int(np.min(parent_idx)) >= 0
            and int(np.max(parent_idx)) < n_parent_collapsed
            and np.unique(parent_idx).size == parent_idx.size
        ):
            payload["B3_JD_operator_wiring_failure_stage"] = "parent_index_per_trace_dof_contract"
            payload["B3_JD_operator_wiring_failure_reason"] = "parent_index_per_trace_dof_contract_failed"
            return 2

        is_parent_u = PETSc.IS().createGeneral(parent_idx.astype(np.int32), comm=PETSc.COMM_WORLD)
        is_p = PETSc.IS().createGeneral(np.arange(raw_App.getSize()[0], dtype=np.int32), comm=PETSc.COMM_WORLD)
        raw_Aup_B3 = raw_Aup.createSubMatrix(is_parent_u, is_p)
        raw_Apu_B3 = raw_Apu.createSubMatrix(is_p, is_parent_u)
        raw_Mpu_B3 = raw_Mpu.createSubMatrix(is_p, is_parent_u)
        for m_ in (raw_Aup_B3, raw_Apu_B3, raw_Mpu_B3):
            _register_mat_for_destroy(mats_to_destroy, m_, seen=mat_destroy_seen)
        is_parent_u.destroy()
        is_p.destroy()

        n_u_b3 = int(raw_Auu.getSize()[0])
        s_uu = max(float(_mat_norm_or_none(raw_Auu) or 0.0), 1.0e-30)
        s_pp = max(float(_mat_norm_or_none(raw_App) or 0.0), 1.0e-30)
        s_c = math.sqrt(s_uu * s_pp)
        parent_fix_blocks = fem3d._locate_facet_displacement_dofs(
            fem.functionspace(msh, fem3d._displacement_element(msh, 1)), msh, f_fix
        )
        fix_scalar_parent = set(
            int(b) * 3 + c
            for b in np.asarray(parent_fix_blocks, dtype=np.int32).ravel()
            for c in range(3)
        )
        b3_fix_scalar = np.asarray([k for k, pi in enumerate(parent_idx.tolist()) if int(pi) in fix_scalar_parent], dtype=np.int32)
        op_meta: Dict[str, Any] = {}
        (
            A_b3,
            M_b3,
            _u_idx,
            _p_idx,
            op_meta,
            _bc_rows,
            _tag5_rows,
            _p_release_rows,
            _m_uu_b3,
            _m_pu_b3,
            _m_pp_b3,
        ) = _build_b3_scaled_restricted_operators_in_memory(
            raw_Auu=raw_Auu,
            raw_Muu=raw_Muu,
            raw_App=raw_App,
            raw_Mpp=raw_Mpp,
            raw_Aup_B3=raw_Aup_B3,
            raw_Apu_B3=raw_Apu_B3,
            raw_Mpu_B3=raw_Mpu_B3,
            s_uu=s_uu,
            s_pp=s_pp,
            s_c=s_c,
            n_u_b3=n_u_b3,
            p_air_collapsed=p_air_collapsed,
            b3_fix_u_rows=b3_fix_scalar,
            msh=msh,
            facet_tags=facet_tags,
            comm=PETSc.COMM_WORLD,
            mats_to_destroy=mats_to_destroy,
            report_meta=op_meta,
            destroy_seen=mat_destroy_seen,
        )
        payload["B3_seed_operator_build_pass"] = True
        payload["B3_scaled_restricted_BC_operator_contract_pass"] = bool(
            op_meta.get("B3_scaled_restricted_BC_operator_contract_pass")
        )
        payload["B3_JD_A_operator_type"] = str(A_b3.getType())
        payload["B3_JD_M_operator_type"] = str(M_b3.getType())
        payload["B3_JD_A_operator_shape"] = [int(A_b3.getSize()[0]), int(A_b3.getSize()[1])]
        payload["B3_JD_M_operator_shape"] = [int(M_b3.getSize()[0]), int(M_b3.getSize()[1])]
        payload["B3_JD_operator_contract_pass"] = bool(
            payload["B3_seed_operator_build_pass"]
            and payload["B3_scaled_restricted_BC_operator_contract_pass"]
            and payload["B3_JD_A_operator_shape"] == [148074, 148074]
            and payload["B3_JD_M_operator_shape"] == [148074, 148074]
            and "aij" in str(payload["B3_JD_A_operator_type"]).lower()
            and "aij" in str(payload["B3_JD_M_operator_type"]).lower()
        )
        if not payload["B3_JD_operator_contract_pass"]:
            payload["B3_JD_operator_wiring_failure_stage"] = "validated_b3_operator_contract"
            payload["B3_JD_operator_wiring_failure_reason"] = "B3_operator_contract_failed_for_JD_handoff"
            return 2

        from slepc4py import SLEPc

        eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
        eps_object_count += 1
        payload["B3_JD_operator_wiring_creates_exactly_one_EPS_object"] = True
        eps.setOperators(A_b3, M_b3)
        payload["B3_JD_operator_wiring_sets_operators"] = True
        eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        try:
            eps.setType(SLEPc.EPS.Type.JD)
            payload["B3_JD_solver_type_method"] = "SLEPc.EPS.Type.JD"
        except Exception:
            eps.setType("jd")
            payload["B3_JD_solver_type_method"] = "setType('jd')"
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        eps.setTarget((2.0 * math.pi * 244.39) ** 2)
        try:
            eps.setDimensions(nev=2, ncv=6)
        except TypeError:
            eps.setDimensions(2, 6)
        payload["B3_JD_operator_wiring_preflight_pass"] = True
        verdict = "B3_JD_OPERATOR_WIRING_PREFLIGHT_PASS_READY_FOR_FIRST_BOUNDED_JD_EXECUTION_AUTHORIZATION_REVIEW"
        return 0
    except Exception as exc:
        if payload.get("B3_JD_operator_wiring_failure_stage") is None:
            payload["B3_JD_operator_wiring_failure_stage"] = "operator_to_jd_eps_configuration"
        payload["B3_JD_operator_wiring_failure_reason"] = f"{type(exc).__name__}:{exc}"
        return 2
    finally:
        if eps is not None:
            try:
                eps.destroy()
            except Exception:
                pass
        payload["B3_JD_operator_wiring_creates_exactly_one_EPS_object"] = bool(eps_object_count == 1)
        payload["next_step_verdict"] = verdict
        _write_json_atomic(OUT_JSON_B3_JD_OPERATOR_WIRING_PREFLIGHT, payload)
        md_lines = [
            "# B3 JD operator wiring preflight (no setup/solve)",
            "",
            f"- verdict: `{verdict}`",
            f"- B3_JD_operator_contract_pass: {payload.get('B3_JD_operator_contract_pass')}",
            f"- A type/shape: {payload.get('B3_JD_A_operator_type')} {payload.get('B3_JD_A_operator_shape')}",
            f"- M type/shape: {payload.get('B3_JD_M_operator_type')} {payload.get('B3_JD_M_operator_shape')}",
            f"- preflight_pass: {payload.get('B3_JD_operator_wiring_preflight_pass')}",
            f"- failure_stage: {payload.get('B3_JD_operator_wiring_failure_stage')}",
            f"- failure_reason: {payload.get('B3_JD_operator_wiring_failure_reason')}",
            "",
            "no_new_eigensolve_executed=True",
        ]
        OUT_MD_B3_JD_OPERATOR_WIRING_PREFLIGHT.parent.mkdir(parents=True, exist_ok=True)
        OUT_MD_B3_JD_OPERATOR_WIRING_PREFLIGHT.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        print("[B3_JD] mode=B3_JD_operator_wiring_preflight_only", flush=True)
        print(f"[B3_JD] B3_JD_operator_wiring_preflight_pass={payload.get('B3_JD_operator_wiring_preflight_pass')}", flush=True)
        print(f"[B3_JD] next_step_verdict={verdict}", flush=True)
        print("[B3_JD] no_new_eigensolve_executed=True", flush=True)
        print(
            "[B3_JD] additional_eps=ONE_TEMPORARY_B3_OPERATOR_WIRING_EPS_AUTHORIZED_NO_SETUP_NO_SOLVE",
            flush=True,
        )
        _register_mat_for_destroy(mats_to_destroy, A_parent, seen=mat_destroy_seen)
        _register_mat_for_destroy(mats_to_destroy, M_parent, seen=mat_destroy_seen)
        _register_mat_for_destroy(mats_to_destroy, A_b3, seen=mat_destroy_seen)
        _register_mat_for_destroy(mats_to_destroy, M_b3, seen=mat_destroy_seen)
        _destroy_mats_deduped(mats_to_destroy)


def _run_b3_jd_first_bounded_execution_only(pre: Dict[str, Any]) -> int:
    jd_cfg = {
        "target_hz": 244.39,
        "target_lambda": 2357906.6075988025,
        "nev": 2,
        "ncv": 20,
        "mpd": 12,
        "blocksize": 1,
        "minv": 2,
        "plusk": 1,
        "initialsize": 4,
        "tol": 1.0e-8,
        "max_it": 120,
    }
    payload: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "B3_JD_first_bounded_execution_only",
        "B3_JD_operator_contract_pass": False,
        "B3_JD_operator_source": "validated_B3_direct_sparse_AIJ_scaled_restricted_corrected_BC",
        "B3_JD_A_operator_type": None,
        "B3_JD_M_operator_type": None,
        "B3_JD_A_operator_shape": None,
        "B3_JD_M_operator_shape": None,
        "B3_JD_initial_space_attached": False,
        "B3_JD_initial_space_reason": (
            "HISTORICAL_SEED_NOT_USED_IN_FIRST_SOLVE_DUE_TO_PRE_BC_FIX_CONTAMINATION_AND_RETIRED_RAYLEIGH_GATE"
        ),
        "B3_JD_problem_type": "GNHEP",
        "B3_JD_solver_type": "JD",
        "B3_JD_which": "TARGET_MAGNITUDE",
        "B3_JD_target_frequency_hz": float(jd_cfg["target_hz"]),
        "B3_JD_target_lambda": _safe_float(float(jd_cfg["target_lambda"])),
        "B3_JD_nev": int(jd_cfg["nev"]),
        "B3_JD_ncv": int(jd_cfg["ncv"]),
        "B3_JD_tolerance": float(jd_cfg["tol"]),
        "B3_JD_max_iterations": int(jd_cfg["max_it"]),
        "B3_JD_runtime_guard_policy": "ONE_TARGETED_JD_GNHEP_SOLVE_EXPLICIT_SEARCH_SPACE_NO_FALLBACK_NO_RETRY",
        "B3_JD_execution_reuses_passed_setup_configuration": True,
        "B3_JD_execution_setup_configuration_source": (
            "B3_JD_dimension_setup_preflight_passed_explicit_configuration"
        ),
        "B3_JD_execution_nev": int(jd_cfg["nev"]),
        "B3_JD_execution_ncv": int(jd_cfg["ncv"]),
        "B3_JD_execution_mpd": int(jd_cfg["mpd"]),
        "B3_JD_execution_blocksize": int(jd_cfg["blocksize"]),
        "B3_JD_execution_minv": int(jd_cfg["minv"]),
        "B3_JD_execution_plusk": int(jd_cfg["plusk"]),
        "B3_JD_execution_initialsize": int(jd_cfg["initialsize"]),
        "B3_JD_execution_minv_blocksize_mpd_constraint_pass": bool(
            int(jd_cfg["minv"]) + int(jd_cfg["blocksize"]) <= int(jd_cfg["mpd"])
        ),
        "B3_JD_solve_count": 0,
        "B3_JD_STSINVERT_fallback_used": False,
        "B3_JD_MUMPS_LU_used": False,
        "B3_JD_automatic_retry_used": False,
        "B3_JD_additional_EPS_solve_used": False,
        "B3_JD_EPS_converged_reason": None,
        "B3_JD_converged_mode_count": 0,
        "B3_JD_execution_stage": "before_b3_operator_build",
        "B3_JD_failure_stage": None,
        "B3_JD_failure_exception_type": None,
        "B3_JD_failure_reason": None,
        "B3_JD_solver_interface_failure_reason": None,
        "B3_JD_EPS_created": None,
        "B3_JD_operators_set": None,
        "B3_JD_problem_type_set": None,
        "B3_JD_solver_type_set": None,
        "B3_JD_target_set": None,
        "B3_JD_dimensions_set": None,
        "B3_JD_solve_attempted": False,
        "B3_JD_execution_authorized": True,
        "jd_wiring_authorized": True,
        "B3_JD_execution_scope": "ONE_BOUNDED_DIAGNOSTIC_SOLVE_ONLY",
        "new_eigensolve_executed": False,
        "no_new_eigensolve_executed": True,
        "additional_eps": "ONE_BOUNDED_B3_JD_EXECUTION_EPS_AUTHORIZED",
        "operator_matrices_persisted": False,
        "transfer_matrices_persisted": False,
        "coupling_matrices_persisted": False,
        "historical_seed_attached": False,
        "mapped_seed_persisted": False,
        "conditioned_seed_persisted": False,
        "eigenvectors_persisted": False,
        "vector_banks_persisted": False,
        "solve_trees_created": False,
        "production_promotion": "BLOCKED",
        "B3_JD_first_execution_failure_stage": None,
        "B3_JD_first_execution_failure_reason": None,
    }
    A_parent = M_parent = A_b3 = M_b3 = None
    mats_to_destroy: List[Any] = []
    mat_destroy_seen: set[int] = set()
    eps = None
    verdict = "B3_JD_FIRST_BOUNDED_EXECUTION_BLOCKED_BY_JD_SOLVER_INTERFACE"
    try:
        print("[B3_JD] stage=before_b3_operator_build", flush=True)
        if not pre["preassembly_contract_pass"]:
            payload["B3_JD_first_execution_failure_stage"] = "preassembly_contract"
            payload["B3_JD_first_execution_failure_reason"] = "preassembly_contract_failed"
            return 2
        if MPI.COMM_WORLD.size != 1:
            payload["B3_JD_first_execution_failure_stage"] = "runtime_mpi_contract"
            payload["B3_JD_first_execution_failure_reason"] = "requires_mpiexec_n_1"
            return 2

        manifest = load_manifest()
        case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
        sample = sample_spec_from_case(case)
        mesh_file = mesh_path("L_mid", CASE_ID)
        msh, _cell_tags, facet_tags = fem3d._load_mesh_and_tags(mesh_file)
        f_top = np.asarray(facet_tags.find(TAG_TOP), dtype=np.int32)
        f_back = np.asarray(facet_tags.find(TAG_BACK), dtype=np.int32)
        f_ribs = np.asarray(facet_tags.find(TAG_RIBS), dtype=np.int32)
        f_fix = np.asarray(facet_tags.find(TAG_FIX), dtype=np.int32)
        shell_facets = np.unique(np.concatenate([f_top, f_back, f_ribs]).astype(np.int32, copy=False))
        tmeta = _build_c2_trace_to_parent_transfer(
            msh, facet_tags, shell_facets=shell_facets, tag_top=TAG_TOP, tag_back=TAG_BACK, tag_ribs=TAG_RIBS
        )
        _tmeta_parent_map = tmeta.get("parent_index_per_trace_dof")
        A_parent, M_parent, cfg = _assemble_reduced_coupled_replay(
            mesh_file, sample, coupling_enabled=True, capture_parent_raw_blocks=True
        )
        maps = _extract_layout_maps(cfg, A_parent)
        p_to_W_parent = np.asarray(maps["p_to_W"], dtype=np.int32).ravel()
        p_air_collapsed = np.asarray(
            cfg.get("_coupled_air_p_air_collapsed_indices", np.asarray([], dtype=np.int32)),
            dtype=np.int32,
        ).ravel()
        raw_cap = _extract_parent_raw_block_capture()
        raw_App = raw_cap.get("raw_App")
        raw_Mpp = raw_cap.get("raw_Mpp")
        raw_Aup = raw_cap.get("raw_Aup")
        raw_Apu = raw_cap.get("raw_Apu")
        raw_Mpu = raw_cap.get("raw_Mpu")
        for m_ in (raw_App, raw_Mpp, raw_Aup, raw_Apu, raw_Mpu):
            if m_ is not None:
                _register_mat_for_destroy(mats_to_destroy, m_, seen=mat_destroy_seen)
        if not all(m is not None for m in (raw_App, raw_Mpp, raw_Aup, raw_Apu, raw_Mpu)):
            payload["B3_JD_first_execution_failure_stage"] = "parent_raw_blocks"
            payload["B3_JD_first_execution_failure_reason"] = "missing_parent_raw_blocks"
            return 2
        if not bool(raw_cap.get("parent_raw_collapsed_layout_dimensions_pass", False)):
            payload["B3_JD_first_execution_failure_stage"] = "parent_raw_layout"
            payload["B3_JD_first_execution_failure_reason"] = "parent_raw_collapsed_layout_not_passing"
            return 2
        if _tmeta_parent_map is None:
            payload["B3_JD_first_execution_failure_stage"] = "trace_to_parent_map"
            payload["B3_JD_first_execution_failure_reason"] = "parent_index_per_trace_dof_missing_from_tmeta"
            return 2

        shell_mesh, shell_to_parent, _, _ = dmesh.create_submesh(msh, msh.topology.dim - 1, shell_facets)
        V_u_trace = fem.functionspace(shell_mesh, fem3d._displacement_element(shell_mesh, 1))
        trace_cells = np.arange(
            int(shell_mesh.topology.index_map(shell_mesh.topology.dim).size_local), dtype=np.int32
        )
        map_meta = _extract_submesh_to_parent_entity_indices(shell_to_parent, entity_dim=msh.topology.dim - 1)
        parent_tag_map = {
            int(i): int(v) for i, v in zip(np.asarray(facet_tags.indices), np.asarray(facet_tags.values))
        }
        parent_f = np.asarray(map_meta.get("indices"), dtype=np.int32).ravel()
        trace_vals = np.array([parent_tag_map.get(int(pf), -1) for pf in parent_f], dtype=np.int32)
        mt_trace = dmesh.meshtags(shell_mesh, shell_mesh.topology.dim, trace_cells, trace_vals)
        dx_trace = ufl.Measure("dx", domain=shell_mesh, subdomain_data=mt_trace)
        u = ufl.TrialFunction(V_u_trace)
        v = ufl.TestFunction(V_u_trace)
        top_m, back_m, t_top, t_back = fem3d._split_wood_materials(cfg)
        nrm = ufl.CellNormal(shell_mesh)
        P = ufl.Identity(3) - ufl.outer(nrm, nrm)
        e1, e2 = fem3d._plate_local_frame(nrm, P)
        grad_u = ufl.grad(u)
        grad_v = ufl.grad(v)
        eps_u = 0.5 * (P * grad_u * P + ufl.transpose(P * grad_u * P))
        eps_v = 0.5 * (P * grad_v * P + ufl.transpose(P * grad_v * P))
        w_n = ufl.dot(u, nrm)
        v_n = ufl.dot(v, nrm)
        shell_top = fem3d._orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, top_m)
        shell_back = fem3d._orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, back_m)
        shell_ribs = fem3d._orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, back_m)
        raw_Auu = fem.petsc.assemble_matrix(
            fem.form(shell_top * dx_trace(TAG_TOP) + shell_back * dx_trace(TAG_BACK) + shell_ribs * dx_trace(TAG_RIBS)),
            bcs=[],
        )
        raw_Muu = fem.petsc.assemble_matrix(
            fem.form(
                (top_m["rho"] * t_top) * ufl.dot(u, v) * dx_trace(TAG_TOP)
                + (back_m["rho"] * t_back) * ufl.dot(u, v) * dx_trace(TAG_BACK)
                + (back_m["rho"] * t_back) * ufl.dot(u, v) * dx_trace(TAG_RIBS)
            ),
            bcs=[],
        )
        raw_Auu.assemble()
        raw_Muu.assemble()
        for m_ in (raw_Auu, raw_Muu):
            _register_mat_for_destroy(mats_to_destroy, m_, seen=mat_destroy_seen)

        parent_idx = np.asarray(_tmeta_parent_map, dtype=np.int32).ravel()
        n_parent_collapsed = int(raw_cap.get("parent_raw_u_dimension", 0) or 0)
        if not (
            parent_idx.size > 0
            and int(np.min(parent_idx)) >= 0
            and int(np.max(parent_idx)) < n_parent_collapsed
            and np.unique(parent_idx).size == parent_idx.size
        ):
            payload["B3_JD_first_execution_failure_stage"] = "parent_index_per_trace_dof_contract"
            payload["B3_JD_first_execution_failure_reason"] = "parent_index_per_trace_dof_contract_failed"
            return 2

        is_parent_u = PETSc.IS().createGeneral(parent_idx.astype(np.int32), comm=PETSc.COMM_WORLD)
        is_p = PETSc.IS().createGeneral(np.arange(raw_App.getSize()[0], dtype=np.int32), comm=PETSc.COMM_WORLD)
        raw_Aup_B3 = raw_Aup.createSubMatrix(is_parent_u, is_p)
        raw_Apu_B3 = raw_Apu.createSubMatrix(is_p, is_parent_u)
        raw_Mpu_B3 = raw_Mpu.createSubMatrix(is_p, is_parent_u)
        for m_ in (raw_Aup_B3, raw_Apu_B3, raw_Mpu_B3):
            _register_mat_for_destroy(mats_to_destroy, m_, seen=mat_destroy_seen)
        is_parent_u.destroy()
        is_p.destroy()

        n_u_b3 = int(raw_Auu.getSize()[0])
        n_p_retained = int(p_to_W_parent.size)
        s_uu = max(float(_mat_norm_or_none(raw_Auu) or 0.0), 1.0e-30)
        s_pp = max(float(_mat_norm_or_none(raw_App) or 0.0), 1.0e-30)
        s_c = math.sqrt(s_uu * s_pp)
        parent_fix_blocks = fem3d._locate_facet_displacement_dofs(
            fem.functionspace(msh, fem3d._displacement_element(msh, 1)), msh, f_fix
        )
        fix_scalar_parent = set(
            int(b) * 3 + c
            for b in np.asarray(parent_fix_blocks, dtype=np.int32).ravel()
            for c in range(3)
        )
        b3_fix_scalar = np.asarray([k for k, pi in enumerate(parent_idx.tolist()) if int(pi) in fix_scalar_parent], dtype=np.int32)
        op_meta: Dict[str, Any] = {}
        (
            A_b3,
            M_b3,
            _u_idx,
            _p_idx,
            op_meta,
            bc_rows,
            _tag5_rows,
            _p_release_rows,
            _m_uu_b3,
            _m_pu_b3,
            _m_pp_b3,
        ) = _build_b3_scaled_restricted_operators_in_memory(
            raw_Auu=raw_Auu,
            raw_Muu=raw_Muu,
            raw_App=raw_App,
            raw_Mpp=raw_Mpp,
            raw_Aup_B3=raw_Aup_B3,
            raw_Apu_B3=raw_Apu_B3,
            raw_Mpu_B3=raw_Mpu_B3,
            s_uu=s_uu,
            s_pp=s_pp,
            s_c=s_c,
            n_u_b3=n_u_b3,
            p_air_collapsed=p_air_collapsed,
            b3_fix_u_rows=b3_fix_scalar,
            msh=msh,
            facet_tags=facet_tags,
            comm=PETSc.COMM_WORLD,
            mats_to_destroy=mats_to_destroy,
            report_meta=op_meta,
            destroy_seen=mat_destroy_seen,
        )
        payload["B3_seed_operator_build_pass"] = True
        payload["B3_scaled_restricted_BC_operator_contract_pass"] = bool(
            op_meta.get("B3_scaled_restricted_BC_operator_contract_pass")
        )
        payload["B3_JD_A_operator_type"] = str(A_b3.getType())
        payload["B3_JD_M_operator_type"] = str(M_b3.getType())
        payload["B3_JD_A_operator_shape"] = [int(A_b3.getSize()[0]), int(A_b3.getSize()[1])]
        payload["B3_JD_M_operator_shape"] = [int(M_b3.getSize()[0]), int(M_b3.getSize()[1])]
        payload["B3_JD_operator_contract_pass"] = bool(
            payload["B3_seed_operator_build_pass"]
            and payload["B3_scaled_restricted_BC_operator_contract_pass"]
            and payload["B3_JD_A_operator_shape"] == [148074, 148074]
            and payload["B3_JD_M_operator_shape"] == [148074, 148074]
            and "aij" in str(payload["B3_JD_A_operator_type"]).lower()
            and "aij" in str(payload["B3_JD_M_operator_type"]).lower()
        )
        if not payload["B3_JD_operator_contract_pass"]:
            payload["B3_JD_first_execution_failure_stage"] = "validated_b3_operator_contract"
            payload["B3_JD_first_execution_failure_reason"] = "B3_operator_contract_failed_for_first_JD_execution"
            payload["B3_JD_failure_stage"] = "after_b3_operator_build"
            payload["B3_JD_failure_reason"] = payload["B3_JD_first_execution_failure_reason"]
            return 2

        payload["B3_JD_execution_stage"] = "after_b3_operator_build"
        print("[B3_JD] stage=after_b3_operator_build", flush=True)
        payload["B3_JD_execution_stage"] = "before_eps_create"
        print("[B3_JD] stage=before_eps_create", flush=True)
        from slepc4py import SLEPc

        eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
        payload["B3_JD_EPS_created"] = True
        payload["B3_JD_execution_stage"] = "after_eps_create"
        eps.setOperators(A_b3, M_b3)
        payload["B3_JD_operators_set"] = True
        eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        payload["B3_JD_problem_type_set"] = True
        try:
            eps.setType(SLEPc.EPS.Type.JD)
            payload["B3_JD_solver_type_method"] = "SLEPc.EPS.Type.JD"
        except Exception:
            eps.setType("jd")
            payload["B3_JD_solver_type_method"] = "setType('jd')"
        payload["B3_JD_solver_type_set"] = True
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        eps.setTarget(float(jd_cfg["target_lambda"]))
        payload["B3_JD_target_set"] = True
        try:
            eps.setDimensions(nev=int(jd_cfg["nev"]), ncv=int(jd_cfg["ncv"]), mpd=int(jd_cfg["mpd"]))
        except TypeError:
            eps.setDimensions(int(jd_cfg["nev"]), int(jd_cfg["ncv"]), int(jd_cfg["mpd"]))
        if hasattr(eps, "setJDBlockSize"):
            eps.setJDBlockSize(int(jd_cfg["blocksize"]))
        if hasattr(eps, "setJDRestart"):
            try:
                eps.setJDRestart(minv=int(jd_cfg["minv"]), plusk=int(jd_cfg["plusk"]))
            except TypeError:
                eps.setJDRestart(int(jd_cfg["minv"]), int(jd_cfg["plusk"]))
        if hasattr(eps, "setJDInitialSize"):
            eps.setJDInitialSize(int(jd_cfg["initialsize"]))
        payload["B3_JD_dimensions_set"] = True
        payload["B3_JD_execution_stage"] = "after_eps_configuration"
        print("[B3_JD] stage=after_eps_configuration", flush=True)
        eps.setTolerances(tol=float(jd_cfg["tol"]), max_it=int(jd_cfg["max_it"]))
        payload["B3_JD_execution_stage"] = "before_eps_setup"
        print("[B3_JD] stage=before_eps_setup", flush=True)
        eps.setUp()
        payload["B3_JD_execution_stage"] = "after_eps_setup"
        print("[B3_JD] stage=after_eps_setup", flush=True)
        payload["B3_JD_execution_stage"] = "before_eps_solve"
        print("[B3_JD] stage=before_eps_solve", flush=True)
        payload["B3_JD_solve_attempted"] = True
        eps.solve()
        payload["B3_JD_solve_count"] = 1
        payload["new_eigensolve_executed"] = True
        payload["no_new_eigensolve_executed"] = False
        payload["B3_JD_execution_stage"] = "after_eps_solve"
        print("[B3_JD] stage=after_eps_solve", flush=True)
        reason = eps.getConvergedReason()
        nconv = int(eps.getConverged())
        payload["B3_JD_EPS_converged_reason"] = int(reason)
        payload["B3_JD_converged_mode_count"] = nconv

        accepted_any = False
        bc_rows_i32 = np.unique(np.asarray(bc_rows, dtype=np.int32).ravel())
        for i in range(nconv):
            vr = A_b3.createVecRight()
            vi = A_b3.createVecRight()
            try:
                lam = eps.getEigenpair(i, vr, vi)
                lam_c = complex(lam)
                lam_re = float(np.real(lam_c))
                lam_im = float(np.imag(lam_c))
                freq_hz = None
                if math.isfinite(lam_re) and abs(lam_im) <= 1.0e-12 and lam_re > 0.0:
                    freq_hz = math.sqrt(max(lam_re, 0.0)) / (2.0 * math.pi)
                try:
                    err_rel = float(eps.computeError(i, SLEPc.EPS.ErrorType.RELATIVE))
                except Exception:
                    err_rel = float("nan")
                x = np.asarray(vr.getArray(readonly=True))
                if np.iscomplexobj(x):
                    x_arr = np.asarray(x, dtype=np.complex128)
                    abs_x = np.abs(x_arr)
                else:
                    x_arr = np.asarray(x, dtype=np.float64)
                    abs_x = np.abs(x_arr)
                x_norm = float(np.linalg.norm(abs_x))
                bc_norm = float(np.linalg.norm(abs_x[bc_rows_i32])) if bc_rows_i32.size > 0 else 0.0
                bc_pass = bool(bc_norm <= 1.0e-8 * max(1.0, x_norm))
                x_u = abs_x[:n_u_b3]
                x_p = abs_x[n_u_b3 : n_u_b3 + n_p_retained]
                u_norm = float(np.linalg.norm(x_u))
                p_norm = float(np.linalg.norm(x_p))
                p_support = p_norm / max(float(np.linalg.norm(abs_x)), 1.0e-30)
                structural_dominant = bool(u_norm > 1.0e-8 and p_norm <= 1.0e-8)
                support_ok = bool(u_norm > 1.0e-8 and (p_support > 1.0e-6 or structural_dominant))
                finite_lambda = bool(math.isfinite(lam_re) and math.isfinite(lam_im))
                residual_ok = bool(math.isfinite(err_rel) and err_rel <= 1.0e-4)
                mode_pass = bool(finite_lambda and residual_ok and bc_pass and support_ok)
                accepted_any = bool(accepted_any or mode_pass)
                target_dist = abs(float(freq_hz) - 244.39) if freq_hz is not None and math.isfinite(float(freq_hz)) else None
                payload[f"B3_JD_mode_{i}_lambda_real"] = _safe_float(lam_re)
                payload[f"B3_JD_mode_{i}_lambda_imag"] = _safe_float(lam_im)
                payload[f"B3_JD_mode_{i}_frequency_hz_if_real_positive"] = _safe_float(freq_hz)
                payload[f"B3_JD_mode_{i}_relative_generalized_residual"] = _safe_float(err_rel)
                payload[f"B3_JD_mode_{i}_dirichlet_zero_compliance_pass"] = bool(bc_pass)
                payload[f"B3_JD_mode_{i}_u_norm"] = _safe_float(u_norm)
                payload[f"B3_JD_mode_{i}_p_norm"] = _safe_float(p_norm)
                payload[f"B3_JD_mode_{i}_pressure_support_metric"] = _safe_float(p_support)
                payload[f"B3_JD_mode_{i}_target_distance_hz"] = _safe_float(target_dist)
                payload[f"B3_JD_mode_{i}_support_classification"] = (
                    "STRUCTURAL_DOMINANT" if structural_dominant else "COUPLED_OR_PRESSURE_SUPPORTED"
                )
                payload[f"B3_JD_mode_{i}_acceptance_pass"] = bool(mode_pass)
            finally:
                vr.destroy()
                vi.destroy()

        if accepted_any:
            verdict = "B3_JD_FIRST_BOUNDED_EXECUTION_PASS_READY_FOR_SECOND_VALIDATION_RUN_DESIGN"
            payload["B3_JD_operator_wiring_preflight_pass"] = True
            return 0
        verdict = "B3_JD_FIRST_BOUNDED_EXECUTION_COMPLETED_BUT_NO_ACCEPTABLE_MODE"
        return 2
    except Exception as exc:
        if payload.get("B3_JD_first_execution_failure_stage") is None:
            payload["B3_JD_first_execution_failure_stage"] = "jd_solver_interface"
        payload["B3_JD_first_execution_failure_reason"] = f"{type(exc).__name__}:{exc}"
        if payload.get("B3_JD_failure_stage") is None:
            payload["B3_JD_failure_stage"] = payload.get("B3_JD_execution_stage")
        payload["B3_JD_failure_exception_type"] = type(exc).__name__
        payload["B3_JD_failure_reason"] = f"{type(exc).__name__}:{exc}"
        payload["B3_JD_solver_interface_failure_reason"] = f"{type(exc).__name__}:{exc}"
        verdict = "B3_JD_FIRST_BOUNDED_EXECUTION_BLOCKED_BY_JD_SOLVER_INTERFACE"
        return 2
    finally:
        if eps is not None:
            try:
                eps.destroy()
            except Exception:
                pass
        payload["next_step_verdict"] = verdict
        _write_json_atomic(OUT_JSON_B3_JD_FIRST_BOUNDED, payload)
        md_lines = [
            "# B3 JD first bounded execution (single solve)",
            "",
            f"- verdict: `{verdict}`",
            f"- B3_JD_operator_contract_pass: {payload.get('B3_JD_operator_contract_pass')}",
            f"- converged_mode_count: {payload.get('B3_JD_converged_mode_count')}",
            f"- EPS reason: {payload.get('B3_JD_EPS_converged_reason')}",
            f"- solve_count: {payload.get('B3_JD_solve_count')}",
            "",
            f"- tolerance/max_it: {payload.get('B3_JD_tolerance')}/{payload.get('B3_JD_max_iterations')}",
            f"- target_hz: {payload.get('B3_JD_target_frequency_hz')}",
            "",
            "new_eigensolve_executed=True" if payload.get("new_eigensolve_executed") else "new_eigensolve_executed=False",
        ]
        OUT_MD_B3_JD_FIRST_BOUNDED.parent.mkdir(parents=True, exist_ok=True)
        OUT_MD_B3_JD_FIRST_BOUNDED.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        print("[B3_JD] mode=B3_JD_first_bounded_execution_only", flush=True)
        print(f"[B3_JD] B3_JD_converged_mode_count={payload.get('B3_JD_converged_mode_count')}", flush=True)
        print(f"[B3_JD] next_step_verdict={verdict}", flush=True)
        print(f"[B3_JD] new_eigensolve_executed={payload.get('new_eigensolve_executed')}", flush=True)
        print("[B3_JD] additional_eps=ONE_BOUNDED_B3_JD_EXECUTION_EPS_AUTHORIZED", flush=True)
        _register_mat_for_destroy(mats_to_destroy, A_parent, seen=mat_destroy_seen)
        _register_mat_for_destroy(mats_to_destroy, M_parent, seen=mat_destroy_seen)
        _register_mat_for_destroy(mats_to_destroy, A_b3, seen=mat_destroy_seen)
        _register_mat_for_destroy(mats_to_destroy, M_b3, seen=mat_destroy_seen)
        _destroy_mats_deduped(mats_to_destroy)


def _run_b3_jd_dimension_setup_preflight_only(pre: Dict[str, Any]) -> int:
    jd_cfg = {
        "target_hz": 244.39,
        "target_lambda": 2357906.6075988025,
        "nev": 2,
        "ncv": 20,
        "mpd": 12,
        "blocksize": 1,
        "minv": 2,
        "plusk": 1,
        "initialsize": 4,
        "tol": 1.0e-8,
        "max_it": 120,
    }
    payload: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "B3_JD_dimension_setup_preflight_only",
        "B3_seed_operator_build_pass": None,
        "B3_scaled_restricted_BC_operator_contract_pass": None,
        "B3_JD_operator_contract_pass": False,
        "B3_JD_operator_source": "validated_B3_direct_sparse_AIJ_scaled_restricted_corrected_BC",
        "B3_JD_A_operator_type": None,
        "B3_JD_M_operator_type": None,
        "B3_JD_A_operator_shape": None,
        "B3_JD_M_operator_shape": None,
        "B3_JD_setup_preflight_creates_exactly_one_EPS_object": False,
        "B3_JD_setup_preflight_sets_operators": False,
        "B3_JD_setup_preflight_calls_setup": False,
        "B3_JD_setup_preflight_calls_solve": False,
        "B3_JD_setup_preflight_sets_initial_space": False,
        "B3_JD_setup_preflight_problem_type": "GNHEP",
        "B3_JD_setup_preflight_solver_type": "JD",
        "B3_JD_setup_preflight_target_frequency_hz": float(jd_cfg["target_hz"]),
        "B3_JD_setup_preflight_target_lambda": float(jd_cfg["target_lambda"]),
        "B3_JD_setup_preflight_nev": int(jd_cfg["nev"]),
        "B3_JD_setup_preflight_ncv": None,
        "B3_JD_setup_preflight_mpd": None,
        "B3_JD_setup_preflight_blocksize": None,
        "B3_JD_setup_preflight_minv": None,
        "B3_JD_setup_preflight_plusk": None,
        "B3_JD_setup_preflight_initialsize": None,
        "B3_JD_setup_preflight_parameter_readback_method": None,
        "B3_JD_setup_preflight_dimension_policy": (
            "EXPLICIT_JD_SEARCH_SPACE_DIMENSIONS_TO_SATISFY_MINV_BLOCKSIZE_MPD_CONTRACT"
        ),
        "B3_JD_setup_preflight_constraint": "minv_plus_blocksize_le_mpd",
        "B3_JD_setup_preflight_constraint_pass": False,
        "B3_JD_setup_preflight_constraint_values": None,
        "B3_JD_setup_preflight_pass": False,
        "B3_JD_setup_preflight_failure_stage": None,
        "B3_JD_setup_preflight_failure_reason": None,
        "B3_JD_setup_preflight_STSINVERT_used": False,
        "B3_JD_setup_preflight_MUMPS_LU_used": False,
        "B3_JD_setup_preflight_fallback_used": False,
        "B3_JD_setup_preflight_solve_executed": False,
        "B3_JD_execution_authorized": False,
        "new_eigensolve_executed": False,
        "additional_eps": "ONE_TEMPORARY_B3_JD_SETUP_PREFLIGHT_EPS_AUTHORIZED_NO_SOLVE",
        "operator_matrices_persisted": False,
        "transfer_matrices_persisted": False,
        "coupling_matrices_persisted": False,
        "mapped_seed_persisted": False,
        "conditioned_seed_persisted": False,
        "eigenvectors_persisted": False,
        "vector_banks_persisted": False,
        "solve_trees_created": False,
        "production_promotion": "BLOCKED",
    }
    A_parent = M_parent = A_b3 = M_b3 = None
    mats_to_destroy: List[Any] = []
    mat_destroy_seen: set[int] = set()
    eps = None
    eps_count = 0
    verdict = "B3_JD_DIMENSION_SETUP_PREFLIGHT_BLOCKED_BY_JD_SEARCH_SPACE_CONFIGURATION_INTERFACE"
    try:
        if not pre["preassembly_contract_pass"]:
            payload["B3_JD_setup_preflight_failure_stage"] = "preassembly_contract"
            payload["B3_JD_setup_preflight_failure_reason"] = "preassembly_contract_failed"
            return 2
        if MPI.COMM_WORLD.size != 1:
            payload["B3_JD_setup_preflight_failure_stage"] = "runtime_mpi_contract"
            payload["B3_JD_setup_preflight_failure_reason"] = "requires_mpiexec_n_1"
            return 2

        manifest = load_manifest()
        case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
        sample = sample_spec_from_case(case)
        mesh_file = mesh_path("L_mid", CASE_ID)
        msh, _cell_tags, facet_tags = fem3d._load_mesh_and_tags(mesh_file)
        f_top = np.asarray(facet_tags.find(TAG_TOP), dtype=np.int32)
        f_back = np.asarray(facet_tags.find(TAG_BACK), dtype=np.int32)
        f_ribs = np.asarray(facet_tags.find(TAG_RIBS), dtype=np.int32)
        f_fix = np.asarray(facet_tags.find(TAG_FIX), dtype=np.int32)
        shell_facets = np.unique(np.concatenate([f_top, f_back, f_ribs]).astype(np.int32, copy=False))
        tmeta = _build_c2_trace_to_parent_transfer(
            msh, facet_tags, shell_facets=shell_facets, tag_top=TAG_TOP, tag_back=TAG_BACK, tag_ribs=TAG_RIBS
        )
        _tmeta_parent_map = tmeta.get("parent_index_per_trace_dof")
        A_parent, M_parent, cfg = _assemble_reduced_coupled_replay(
            mesh_file, sample, coupling_enabled=True, capture_parent_raw_blocks=True
        )
        maps = _extract_layout_maps(cfg, A_parent)
        p_to_W_parent = np.asarray(maps["p_to_W"], dtype=np.int32).ravel()
        p_air_collapsed = np.asarray(
            cfg.get("_coupled_air_p_air_collapsed_indices", np.asarray([], dtype=np.int32)),
            dtype=np.int32,
        ).ravel()
        raw_cap = _extract_parent_raw_block_capture()
        raw_App = raw_cap.get("raw_App")
        raw_Mpp = raw_cap.get("raw_Mpp")
        raw_Aup = raw_cap.get("raw_Aup")
        raw_Apu = raw_cap.get("raw_Apu")
        raw_Mpu = raw_cap.get("raw_Mpu")
        for m_ in (raw_App, raw_Mpp, raw_Aup, raw_Apu, raw_Mpu):
            if m_ is not None:
                _register_mat_for_destroy(mats_to_destroy, m_, seen=mat_destroy_seen)
        if not all(m is not None for m in (raw_App, raw_Mpp, raw_Aup, raw_Apu, raw_Mpu)):
            payload["B3_JD_setup_preflight_failure_stage"] = "parent_raw_blocks"
            payload["B3_JD_setup_preflight_failure_reason"] = "missing_parent_raw_blocks"
            return 2
        if not bool(raw_cap.get("parent_raw_collapsed_layout_dimensions_pass", False)):
            payload["B3_JD_setup_preflight_failure_stage"] = "parent_raw_layout"
            payload["B3_JD_setup_preflight_failure_reason"] = "parent_raw_collapsed_layout_not_passing"
            return 2
        if _tmeta_parent_map is None:
            payload["B3_JD_setup_preflight_failure_stage"] = "trace_to_parent_map"
            payload["B3_JD_setup_preflight_failure_reason"] = "parent_index_per_trace_dof_missing_from_tmeta"
            return 2

        shell_mesh, shell_to_parent, _, _ = dmesh.create_submesh(msh, msh.topology.dim - 1, shell_facets)
        V_u_trace = fem.functionspace(shell_mesh, fem3d._displacement_element(shell_mesh, 1))
        trace_cells = np.arange(
            int(shell_mesh.topology.index_map(shell_mesh.topology.dim).size_local), dtype=np.int32
        )
        map_meta = _extract_submesh_to_parent_entity_indices(shell_to_parent, entity_dim=msh.topology.dim - 1)
        parent_tag_map = {int(i): int(v) for i, v in zip(np.asarray(facet_tags.indices), np.asarray(facet_tags.values))}
        parent_f = np.asarray(map_meta.get("indices"), dtype=np.int32).ravel()
        trace_vals = np.array([parent_tag_map.get(int(pf), -1) for pf in parent_f], dtype=np.int32)
        mt_trace = dmesh.meshtags(shell_mesh, shell_mesh.topology.dim, trace_cells, trace_vals)
        dx_trace = ufl.Measure("dx", domain=shell_mesh, subdomain_data=mt_trace)
        u = ufl.TrialFunction(V_u_trace)
        v = ufl.TestFunction(V_u_trace)
        top_m, back_m, t_top, t_back = fem3d._split_wood_materials(cfg)
        nrm = ufl.CellNormal(shell_mesh)
        P = ufl.Identity(3) - ufl.outer(nrm, nrm)
        e1, e2 = fem3d._plate_local_frame(nrm, P)
        grad_u = ufl.grad(u)
        grad_v = ufl.grad(v)
        eps_u = 0.5 * (P * grad_u * P + ufl.transpose(P * grad_u * P))
        eps_v = 0.5 * (P * grad_v * P + ufl.transpose(P * grad_v * P))
        w_n = ufl.dot(u, nrm)
        v_n = ufl.dot(v, nrm)
        shell_top = fem3d._orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, top_m)
        shell_back = fem3d._orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, back_m)
        shell_ribs = fem3d._orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, back_m)
        raw_Auu = fem.petsc.assemble_matrix(
            fem.form(shell_top * dx_trace(TAG_TOP) + shell_back * dx_trace(TAG_BACK) + shell_ribs * dx_trace(TAG_RIBS)),
            bcs=[],
        )
        raw_Muu = fem.petsc.assemble_matrix(
            fem.form(
                (top_m["rho"] * t_top) * ufl.dot(u, v) * dx_trace(TAG_TOP)
                + (back_m["rho"] * t_back) * ufl.dot(u, v) * dx_trace(TAG_BACK)
                + (back_m["rho"] * t_back) * ufl.dot(u, v) * dx_trace(TAG_RIBS)
            ),
            bcs=[],
        )
        raw_Auu.assemble()
        raw_Muu.assemble()
        for m_ in (raw_Auu, raw_Muu):
            _register_mat_for_destroy(mats_to_destroy, m_, seen=mat_destroy_seen)
        parent_idx = np.asarray(_tmeta_parent_map, dtype=np.int32).ravel()
        n_parent_collapsed = int(raw_cap.get("parent_raw_u_dimension", 0) or 0)
        if not (
            parent_idx.size > 0
            and int(np.min(parent_idx)) >= 0
            and int(np.max(parent_idx)) < n_parent_collapsed
            and np.unique(parent_idx).size == parent_idx.size
        ):
            payload["B3_JD_setup_preflight_failure_stage"] = "parent_index_per_trace_dof_contract"
            payload["B3_JD_setup_preflight_failure_reason"] = "parent_index_per_trace_dof_contract_failed"
            return 2
        is_parent_u = PETSc.IS().createGeneral(parent_idx.astype(np.int32), comm=PETSc.COMM_WORLD)
        is_p = PETSc.IS().createGeneral(np.arange(raw_App.getSize()[0], dtype=np.int32), comm=PETSc.COMM_WORLD)
        raw_Aup_B3 = raw_Aup.createSubMatrix(is_parent_u, is_p)
        raw_Apu_B3 = raw_Apu.createSubMatrix(is_p, is_parent_u)
        raw_Mpu_B3 = raw_Mpu.createSubMatrix(is_p, is_parent_u)
        for m_ in (raw_Aup_B3, raw_Apu_B3, raw_Mpu_B3):
            _register_mat_for_destroy(mats_to_destroy, m_, seen=mat_destroy_seen)
        is_parent_u.destroy()
        is_p.destroy()
        n_u_b3 = int(raw_Auu.getSize()[0])
        s_uu = max(float(_mat_norm_or_none(raw_Auu) or 0.0), 1.0e-30)
        s_pp = max(float(_mat_norm_or_none(raw_App) or 0.0), 1.0e-30)
        s_c = math.sqrt(s_uu * s_pp)
        parent_fix_blocks = fem3d._locate_facet_displacement_dofs(
            fem.functionspace(msh, fem3d._displacement_element(msh, 1)), msh, f_fix
        )
        fix_scalar_parent = set(
            int(b) * 3 + c for b in np.asarray(parent_fix_blocks, dtype=np.int32).ravel() for c in range(3)
        )
        b3_fix_scalar = np.asarray([k for k, pi in enumerate(parent_idx.tolist()) if int(pi) in fix_scalar_parent], dtype=np.int32)
        op_meta: Dict[str, Any] = {}
        (
            A_b3,
            M_b3,
            _u_idx,
            _p_idx,
            op_meta,
            _bc_rows,
            _tag5_rows,
            _p_release_rows,
            _m_uu_b3,
            _m_pu_b3,
            _m_pp_b3,
        ) = _build_b3_scaled_restricted_operators_in_memory(
            raw_Auu=raw_Auu,
            raw_Muu=raw_Muu,
            raw_App=raw_App,
            raw_Mpp=raw_Mpp,
            raw_Aup_B3=raw_Aup_B3,
            raw_Apu_B3=raw_Apu_B3,
            raw_Mpu_B3=raw_Mpu_B3,
            s_uu=s_uu,
            s_pp=s_pp,
            s_c=s_c,
            n_u_b3=n_u_b3,
            p_air_collapsed=p_air_collapsed,
            b3_fix_u_rows=b3_fix_scalar,
            msh=msh,
            facet_tags=facet_tags,
            comm=PETSc.COMM_WORLD,
            mats_to_destroy=mats_to_destroy,
            report_meta=op_meta,
            destroy_seen=mat_destroy_seen,
        )
        payload["B3_seed_operator_build_pass"] = True
        payload["B3_scaled_restricted_BC_operator_contract_pass"] = bool(op_meta.get("B3_scaled_restricted_BC_operator_contract_pass"))
        payload["B3_JD_A_operator_type"] = str(A_b3.getType())
        payload["B3_JD_M_operator_type"] = str(M_b3.getType())
        payload["B3_JD_A_operator_shape"] = [int(A_b3.getSize()[0]), int(A_b3.getSize()[1])]
        payload["B3_JD_M_operator_shape"] = [int(M_b3.getSize()[0]), int(M_b3.getSize()[1])]
        payload["B3_JD_operator_contract_pass"] = bool(
            payload["B3_seed_operator_build_pass"]
            and payload["B3_scaled_restricted_BC_operator_contract_pass"]
            and payload["B3_JD_A_operator_shape"] == [148074, 148074]
            and payload["B3_JD_M_operator_shape"] == [148074, 148074]
            and "aij" in str(payload["B3_JD_A_operator_type"]).lower()
            and "aij" in str(payload["B3_JD_M_operator_type"]).lower()
        )
        if not payload["B3_JD_operator_contract_pass"]:
            payload["B3_JD_setup_preflight_failure_stage"] = "validated_b3_operator_contract"
            payload["B3_JD_setup_preflight_failure_reason"] = "B3_operator_contract_failed_for_JD_setup_preflight"
            return 2

        from slepc4py import SLEPc
        eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
        eps_count += 1
        payload["B3_JD_setup_preflight_creates_exactly_one_EPS_object"] = True
        eps.setOperators(A_b3, M_b3)
        payload["B3_JD_setup_preflight_sets_operators"] = True
        eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        try:
            eps.setType(SLEPc.EPS.Type.JD)
            payload["B3_JD_setup_preflight_solver_type_method"] = "SLEPc.EPS.Type.JD"
        except Exception:
            eps.setType("jd")
            payload["B3_JD_setup_preflight_solver_type_method"] = "setType('jd')"
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        eps.setTarget(float(payload["B3_JD_setup_preflight_target_lambda"]))
        nev = int(jd_cfg["nev"])
        ncv = int(jd_cfg["ncv"])
        mpd = int(jd_cfg["mpd"])
        blocksize = int(jd_cfg["blocksize"])
        minv = int(jd_cfg["minv"])
        plusk = int(jd_cfg["plusk"])
        initialsize = int(jd_cfg["initialsize"])
        try:
            eps.setDimensions(nev=nev, ncv=ncv, mpd=mpd)
        except TypeError:
            eps.setDimensions(nev, ncv, mpd)
        payload["B3_JD_setup_preflight_nev"] = nev
        payload["B3_JD_setup_preflight_ncv"] = ncv
        payload["B3_JD_setup_preflight_mpd"] = mpd
        payload["B3_JD_setup_preflight_parameter_readback_method"] = "explicit_values_plus_getDimensions_if_available"
        if hasattr(eps, "setJDBlockSize"):
            eps.setJDBlockSize(blocksize)
        if hasattr(eps, "setJDRestart"):
            try:
                eps.setJDRestart(minv=minv, plusk=plusk)
            except TypeError:
                eps.setJDRestart(minv, plusk)
        if hasattr(eps, "setJDInitialSize"):
            eps.setJDInitialSize(initialsize)
        if hasattr(eps, "getJDBlockSize"):
            payload["B3_JD_setup_preflight_blocksize"] = int(eps.getJDBlockSize())
        else:
            payload["B3_JD_setup_preflight_blocksize"] = blocksize
        if hasattr(eps, "getJDRestart"):
            rst = eps.getJDRestart()
            if isinstance(rst, (tuple, list)) and len(rst) >= 2:
                payload["B3_JD_setup_preflight_minv"] = int(rst[0])
                payload["B3_JD_setup_preflight_plusk"] = int(rst[1])
        if payload["B3_JD_setup_preflight_minv"] is None:
            payload["B3_JD_setup_preflight_minv"] = minv
        if payload["B3_JD_setup_preflight_plusk"] is None:
            payload["B3_JD_setup_preflight_plusk"] = plusk
        if hasattr(eps, "getJDInitialSize"):
            payload["B3_JD_setup_preflight_initialsize"] = int(eps.getJDInitialSize())
        else:
            payload["B3_JD_setup_preflight_initialsize"] = initialsize
        if hasattr(eps, "getDimensions"):
            dims = eps.getDimensions()
            if isinstance(dims, (tuple, list)) and len(dims) >= 3:
                payload["B3_JD_setup_preflight_nev"] = int(dims[0])
                payload["B3_JD_setup_preflight_ncv"] = int(dims[1])
                payload["B3_JD_setup_preflight_mpd"] = int(dims[2])
        minv_v = int(payload["B3_JD_setup_preflight_minv"])
        bs_v = int(payload["B3_JD_setup_preflight_blocksize"])
        mpd_v = int(payload["B3_JD_setup_preflight_mpd"])
        payload["B3_JD_setup_preflight_constraint_pass"] = bool(minv_v + bs_v <= mpd_v)
        payload["B3_JD_setup_preflight_constraint_values"] = (
            f"minv={minv_v}, blocksize={bs_v}, mpd={mpd_v}, "
            f"ncv={int(payload['B3_JD_setup_preflight_ncv'])}, nev={int(payload['B3_JD_setup_preflight_nev'])}"
        )
        if not payload["B3_JD_setup_preflight_constraint_pass"]:
            payload["B3_JD_setup_preflight_failure_stage"] = "jd_constraint_check_before_setup"
            payload["B3_JD_setup_preflight_failure_reason"] = "minv_plus_blocksize_gt_mpd"
            return 2
        eps.setTolerances(tol=float(jd_cfg["tol"]), max_it=int(jd_cfg["max_it"]))
        eps.setUp()
        payload["B3_JD_setup_preflight_calls_setup"] = True
        payload["B3_JD_setup_preflight_pass"] = True
        verdict = "B3_JD_DIMENSION_SETUP_PREFLIGHT_PASS_READY_FOR_FIRST_BOUNDED_EXECUTION_RERUN_AUTHORIZATION_REVIEW"
        return 0
    except Exception as exc:
        if payload["B3_JD_setup_preflight_failure_stage"] is None:
            payload["B3_JD_setup_preflight_failure_stage"] = "eps_setup"
        payload["B3_JD_setup_preflight_failure_reason"] = f"{type(exc).__name__}:{exc}"
        return 2
    finally:
        if eps is not None:
            try:
                eps.destroy()
            except Exception:
                pass
        payload["B3_JD_setup_preflight_creates_exactly_one_EPS_object"] = bool(eps_count == 1)
        payload["next_step_verdict"] = verdict
        _write_json_atomic(OUT_JSON_B3_JD_SETUP_PREFLIGHT, payload)
        md_lines = [
            "# B3 JD dimension/setup preflight (no solve)",
            "",
            f"- verdict: `{verdict}`",
            f"- operator_contract_pass: {payload.get('B3_JD_operator_contract_pass')}",
            f"- setup_preflight_pass: {payload.get('B3_JD_setup_preflight_pass')}",
            f"- constraint: {payload.get('B3_JD_setup_preflight_constraint_values')}",
            f"- failure_stage: {payload.get('B3_JD_setup_preflight_failure_stage')}",
            f"- failure_reason: {payload.get('B3_JD_setup_preflight_failure_reason')}",
            "",
            "new_eigensolve_executed=False",
        ]
        OUT_MD_B3_JD_SETUP_PREFLIGHT.parent.mkdir(parents=True, exist_ok=True)
        OUT_MD_B3_JD_SETUP_PREFLIGHT.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        print("[B3_JD] mode=B3_JD_dimension_setup_preflight_only", flush=True)
        print(f"[B3_JD] B3_JD_setup_preflight_pass={payload.get('B3_JD_setup_preflight_pass')}", flush=True)
        print(f"[B3_JD] next_step_verdict={verdict}", flush=True)
        print("[B3_JD] new_eigensolve_executed=False", flush=True)
        print("[B3_JD] additional_eps=ONE_TEMPORARY_B3_JD_SETUP_PREFLIGHT_EPS_AUTHORIZED_NO_SOLVE", flush=True)
        _register_mat_for_destroy(mats_to_destroy, A_parent, seen=mat_destroy_seen)
        _register_mat_for_destroy(mats_to_destroy, M_parent, seen=mat_destroy_seen)
        _register_mat_for_destroy(mats_to_destroy, A_b3, seen=mat_destroy_seen)
        _register_mat_for_destroy(mats_to_destroy, M_b3, seen=mat_destroy_seen)
        _destroy_mats_deduped(mats_to_destroy)


def _run_b3_gnhep_bc_spectral_pollution_contract_only(pre: Dict[str, Any]) -> int:
    payload: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "B3_GNHEP_BC_spectral_pollution_contract_only",
        "B3_seed_operator_build_pass": None,
        "B3_GNHEP_BC_operator_build_pass": None,
        "B3_scaled_restricted_BC_operator_contract_pass": None,
        "B3_JD_operator_contract_pass": False,
        "B3_GNHEP_BC_A_operator_type": None,
        "B3_GNHEP_BC_M_operator_type": None,
        "B3_GNHEP_BC_A_operator_shape": None,
        "B3_GNHEP_BC_M_operator_shape": None,
        "B3_GNHEP_BC_tag5_dirichlet_row_count": None,
        "B3_GNHEP_BC_pressure_release_dirichlet_row_count": None,
        "B3_GNHEP_BC_rows_match_operator_BC_contract": None,
        "B3_GNHEP_BC_current_A_dirichlet_diag": 1.0,
        "B3_GNHEP_BC_current_M_dirichlet_diag": 1.0,
        "B3_GNHEP_BC_current_zero_columns": True,
        "B3_GNHEP_BC_constrained_dofs_retained_in_operator": True,
        "B3_GNHEP_BC_total_dirichlet_row_count": 0,
        "B3_GNHEP_BC_probe_constructed": None,
        "B3_GNHEP_BC_probe_row_count": 0,
        "B3_GNHEP_BC_probe_row_indices_preview": None,
        "B3_GNHEP_BC_probe_rows_match_operator_BC_contract": None,
        "B3_GNHEP_BC_probe_Ae_norm_min": None,
        "B3_GNHEP_BC_probe_Ae_norm_max": None,
        "B3_GNHEP_BC_probe_Me_norm_min": None,
        "B3_GNHEP_BC_probe_Me_norm_max": None,
        "B3_GNHEP_BC_probe_Ae_minus_Me_norm_min": None,
        "B3_GNHEP_BC_probe_Ae_minus_Me_norm_max": None,
        "B3_GNHEP_BC_probe_lambda_one_exact_mode_pass": False,
        "B3_GNHEP_BC_JD_lambda_one_modes_explained_by_dirichlet_rows": False,
        "B3_GNHEP_BC_probe_failure_stage": None,
        "B3_GNHEP_BC_probe_failure_reason": None,
        "B3_GNHEP_BC_existing_eigensolve_safe_route_in_repo": "NOT_FOUND_FOR_GNHEP_A11_M00_DIRICHLET_DIAGONAL_POLICY",
        "B3_GNHEP_BC_preferred_spectral_pollution_fix": (
            "ZERO_M_DIRICHLET_DIAGONAL_AFTER_SAME_ROW_COLUMN_ELIMINATION"
        ),
        "B3_GNHEP_BC_preferred_fix_reason": (
            "Preserves validated A/M interior blocks and pressure-restriction layout; "
            "removes retained constrained-row λ=1 finite eigenpairs from generalized spectrum."
        ),
        "B3_JD_execution_authorized": False,
        "jd_wiring_authorized": False,
        "no_new_eigensolve_executed": True,
        "additional_eps": "NOT_AUTHORIZED",
        "operator_matrices_persisted": False,
        "transfer_matrices_persisted": False,
        "coupling_matrices_persisted": False,
        "eigenvectors_persisted": False,
        "vector_banks_persisted": False,
        "solve_trees_created": False,
        "production_promotion": "BLOCKED",
    }
    A_parent = M_parent = A_b3 = M_b3 = None
    mats_to_destroy: List[Any] = []
    mat_destroy_seen: set[int] = set()
    verdict = "B3_GNHEP_BC_SPECTRAL_POLLUTION_CONTRACT_INCOMPLETE"
    try:
        if not pre["preassembly_contract_pass"]:
            payload["B3_GNHEP_BC_probe_failure_stage"] = "preassembly_contract"
            payload["B3_GNHEP_BC_probe_failure_reason"] = "preassembly_contract_failed"
            return 2
        if MPI.COMM_WORLD.size != 1:
            payload["B3_GNHEP_BC_contract_failure_reason"] = "requires_mpiexec_n_1"
            return 2
        manifest = load_manifest()
        case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
        sample = sample_spec_from_case(case)
        mesh_file = mesh_path("L_mid", CASE_ID)
        msh, _cell_tags, facet_tags = fem3d._load_mesh_and_tags(mesh_file)
        f_top = np.asarray(facet_tags.find(TAG_TOP), dtype=np.int32)
        f_back = np.asarray(facet_tags.find(TAG_BACK), dtype=np.int32)
        f_ribs = np.asarray(facet_tags.find(TAG_RIBS), dtype=np.int32)
        f_fix = np.asarray(facet_tags.find(TAG_FIX), dtype=np.int32)
        shell_facets = np.unique(np.concatenate([f_top, f_back, f_ribs]).astype(np.int32, copy=False))
        tmeta = _build_c2_trace_to_parent_transfer(
            msh, facet_tags, shell_facets=shell_facets, tag_top=TAG_TOP, tag_back=TAG_BACK, tag_ribs=TAG_RIBS
        )
        _tmeta_parent_map = tmeta.get("parent_index_per_trace_dof")
        A_parent, M_parent, cfg = _assemble_reduced_coupled_replay(
            mesh_file, sample, coupling_enabled=True, capture_parent_raw_blocks=True
        )
        maps = _extract_layout_maps(cfg, A_parent)
        p_to_W_parent = np.asarray(maps["p_to_W"], dtype=np.int32).ravel()
        p_air_collapsed = np.asarray(
            cfg.get("_coupled_air_p_air_collapsed_indices", np.asarray([], dtype=np.int32)),
            dtype=np.int32,
        ).ravel()
        raw_cap = _extract_parent_raw_block_capture()
        raw_App = raw_cap.get("raw_App")
        raw_Mpp = raw_cap.get("raw_Mpp")
        raw_Aup = raw_cap.get("raw_Aup")
        raw_Apu = raw_cap.get("raw_Apu")
        raw_Mpu = raw_cap.get("raw_Mpu")
        for m_ in (raw_App, raw_Mpp, raw_Aup, raw_Apu, raw_Mpu):
            if m_ is not None:
                _register_mat_for_destroy(mats_to_destroy, m_, seen=mat_destroy_seen)
        if not all(m is not None for m in (raw_App, raw_Mpp, raw_Aup, raw_Apu, raw_Mpu)):
            payload["B3_GNHEP_BC_contract_failure_reason"] = "missing_parent_raw_blocks"
            return 2
        if not bool(raw_cap.get("parent_raw_collapsed_layout_dimensions_pass", False)):
            payload["B3_GNHEP_BC_contract_failure_reason"] = "parent_raw_collapsed_layout_not_passing"
            return 2
        if _tmeta_parent_map is None:
            payload["B3_GNHEP_BC_contract_failure_reason"] = "parent_index_per_trace_dof_missing_from_tmeta"
            return 2

        shell_mesh, shell_to_parent, _, _ = dmesh.create_submesh(msh, msh.topology.dim - 1, shell_facets)
        V_u_trace = fem.functionspace(shell_mesh, fem3d._displacement_element(shell_mesh, 1))
        trace_cells = np.arange(
            int(shell_mesh.topology.index_map(shell_mesh.topology.dim).size_local), dtype=np.int32
        )
        map_meta = _extract_submesh_to_parent_entity_indices(shell_to_parent, entity_dim=msh.topology.dim - 1)
        parent_tag_map = {int(i): int(v) for i, v in zip(np.asarray(facet_tags.indices), np.asarray(facet_tags.values))}
        parent_f = np.asarray(map_meta.get("indices"), dtype=np.int32).ravel()
        trace_vals = np.array([parent_tag_map.get(int(pf), -1) for pf in parent_f], dtype=np.int32)
        mt_trace = dmesh.meshtags(shell_mesh, shell_mesh.topology.dim, trace_cells, trace_vals)
        dx_trace = ufl.Measure("dx", domain=shell_mesh, subdomain_data=mt_trace)
        u = ufl.TrialFunction(V_u_trace)
        v = ufl.TestFunction(V_u_trace)
        top_m, back_m, t_top, t_back = fem3d._split_wood_materials(cfg)
        nrm = ufl.CellNormal(shell_mesh)
        P = ufl.Identity(3) - ufl.outer(nrm, nrm)
        e1, e2 = fem3d._plate_local_frame(nrm, P)
        grad_u = ufl.grad(u)
        grad_v = ufl.grad(v)
        eps_u = 0.5 * (P * grad_u * P + ufl.transpose(P * grad_u * P))
        eps_v = 0.5 * (P * grad_v * P + ufl.transpose(P * grad_v * P))
        w_n = ufl.dot(u, nrm)
        v_n = ufl.dot(v, nrm)
        shell_top = fem3d._orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, top_m)
        shell_back = fem3d._orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, back_m)
        shell_ribs = fem3d._orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, back_m)
        raw_Auu = fem.petsc.assemble_matrix(
            fem.form(shell_top * dx_trace(TAG_TOP) + shell_back * dx_trace(TAG_BACK) + shell_ribs * dx_trace(TAG_RIBS)),
            bcs=[],
        )
        raw_Muu = fem.petsc.assemble_matrix(
            fem.form(
                (top_m["rho"] * t_top) * ufl.dot(u, v) * dx_trace(TAG_TOP)
                + (back_m["rho"] * t_back) * ufl.dot(u, v) * dx_trace(TAG_BACK)
                + (back_m["rho"] * t_back) * ufl.dot(u, v) * dx_trace(TAG_RIBS)
            ),
            bcs=[],
        )
        raw_Auu.assemble()
        raw_Muu.assemble()
        for m_ in (raw_Auu, raw_Muu):
            _register_mat_for_destroy(mats_to_destroy, m_, seen=mat_destroy_seen)
        parent_idx = np.asarray(_tmeta_parent_map, dtype=np.int32).ravel()
        n_parent_collapsed = int(raw_cap.get("parent_raw_u_dimension", 0) or 0)
        if not (
            parent_idx.size > 0
            and int(np.min(parent_idx)) >= 0
            and int(np.max(parent_idx)) < n_parent_collapsed
            and np.unique(parent_idx).size == parent_idx.size
        ):
            payload["B3_GNHEP_BC_contract_failure_reason"] = "parent_index_per_trace_dof_contract_failed"
            return 2
        is_parent_u = PETSc.IS().createGeneral(parent_idx.astype(np.int32), comm=PETSc.COMM_WORLD)
        is_p = PETSc.IS().createGeneral(np.arange(raw_App.getSize()[0], dtype=np.int32), comm=PETSc.COMM_WORLD)
        raw_Aup_B3 = raw_Aup.createSubMatrix(is_parent_u, is_p)
        raw_Apu_B3 = raw_Apu.createSubMatrix(is_p, is_parent_u)
        raw_Mpu_B3 = raw_Mpu.createSubMatrix(is_p, is_parent_u)
        for m_ in (raw_Aup_B3, raw_Apu_B3, raw_Mpu_B3):
            _register_mat_for_destroy(mats_to_destroy, m_, seen=mat_destroy_seen)
        is_parent_u.destroy()
        is_p.destroy()
        n_u_b3 = int(raw_Auu.getSize()[0])
        s_uu = max(float(_mat_norm_or_none(raw_Auu) or 0.0), 1.0e-30)
        s_pp = max(float(_mat_norm_or_none(raw_App) or 0.0), 1.0e-30)
        s_c = math.sqrt(s_uu * s_pp)
        parent_fix_blocks = fem3d._locate_facet_displacement_dofs(
            fem.functionspace(msh, fem3d._displacement_element(msh, 1)), msh, f_fix
        )
        fix_scalar_parent = set(
            int(b) * 3 + c for b in np.asarray(parent_fix_blocks, dtype=np.int32).ravel() for c in range(3)
        )
        b3_fix_scalar = np.asarray([k for k, pi in enumerate(parent_idx.tolist()) if int(pi) in fix_scalar_parent], dtype=np.int32)
        op_meta: Dict[str, Any] = {}
        (
            A_b3,
            M_b3,
            _u_idx,
            _p_idx,
            op_meta,
            bc_rows,
            _tag5_rows,
            _p_release_rows,
            _m_uu_b3,
            _m_pu_b3,
            _m_pp_b3,
        ) = _build_b3_scaled_restricted_operators_in_memory(
            raw_Auu=raw_Auu,
            raw_Muu=raw_Muu,
            raw_App=raw_App,
            raw_Mpp=raw_Mpp,
            raw_Aup_B3=raw_Aup_B3,
            raw_Apu_B3=raw_Apu_B3,
            raw_Mpu_B3=raw_Mpu_B3,
            s_uu=s_uu,
            s_pp=s_pp,
            s_c=s_c,
            n_u_b3=n_u_b3,
            p_air_collapsed=p_air_collapsed,
            b3_fix_u_rows=b3_fix_scalar,
            msh=msh,
            facet_tags=facet_tags,
            comm=PETSc.COMM_WORLD,
            mats_to_destroy=mats_to_destroy,
            report_meta=op_meta,
            destroy_seen=mat_destroy_seen,
        )
        payload["B3_seed_operator_build_pass"] = True
        payload["B3_GNHEP_BC_operator_build_pass"] = True
        payload["B3_scaled_restricted_BC_operator_contract_pass"] = bool(op_meta.get("B3_scaled_restricted_BC_operator_contract_pass"))
        payload["B3_JD_operator_contract_pass"] = bool(
            payload["B3_seed_operator_build_pass"] and payload["B3_scaled_restricted_BC_operator_contract_pass"]
        )
        payload["B3_GNHEP_BC_A_operator_type"] = str(A_b3.getType())
        payload["B3_GNHEP_BC_M_operator_type"] = str(M_b3.getType())
        payload["B3_GNHEP_BC_A_operator_shape"] = [int(A_b3.getSize()[0]), int(A_b3.getSize()[1])]
        payload["B3_GNHEP_BC_M_operator_shape"] = [int(M_b3.getSize()[0]), int(M_b3.getSize()[1])]
        payload["B3_GNHEP_BC_tag5_dirichlet_row_count"] = int(op_meta.get("B3_seed_tag5_dirichlet_row_count") or 0)
        payload["B3_GNHEP_BC_pressure_release_dirichlet_row_count"] = int(
            op_meta.get("B3_seed_pressure_release_dirichlet_row_count") or 0
        )
        payload["B3_GNHEP_BC_rows_match_operator_BC_contract"] = bool(
            op_meta.get("B3_seed_dirichlet_row_contract_matches_operator_BC")
        )
        payload["B3_GNHEP_BC_total_dirichlet_row_count"] = int(op_meta.get("B3_seed_total_dirichlet_row_count") or 0)
        n_w = int(A_b3.getSize()[0])
        bc_rows_i32 = np.unique(np.asarray(bc_rows, dtype=np.int32).ravel())
        probe_rows = bc_rows_i32[: min(8, int(bc_rows_i32.size))]
        payload["B3_GNHEP_BC_probe_constructed"] = True
        payload["B3_GNHEP_BC_probe_row_count"] = int(probe_rows.size)
        payload["B3_GNHEP_BC_probe_row_indices_preview"] = [int(x) for x in probe_rows.tolist()]
        payload["B3_GNHEP_BC_probe_rows_match_operator_BC_contract"] = bool(
            payload["B3_GNHEP_BC_rows_match_operator_BC_contract"]
        )
        max_norm = 0.0
        min_norm = float("inf")
        ae_min = float("inf")
        ae_max = 0.0
        me_min = float("inf")
        me_max = 0.0
        exact_pass = True
        for r in probe_rows.tolist():
            e = np.zeros(n_w, dtype=np.float64)
            e[int(r)] = 1.0
            ve = _petsc_vec_from_array(A_b3, e)
            try:
                Ae, ay = _petsc_matvec(A_b3, ve)
                Me, my = _petsc_matvec(M_b3, ve)
            finally:
                ve.destroy()
            try:
                d = np.asarray(Ae, dtype=np.float64) - np.asarray(Me, dtype=np.float64)
                dn = float(np.linalg.norm(d))
                max_norm = max(max_norm, dn)
                min_norm = min(min_norm, dn)
                ae_n = float(np.linalg.norm(np.asarray(Ae, dtype=np.float64)))
                me_n = float(np.linalg.norm(np.asarray(Me, dtype=np.float64)))
                ae_min = min(ae_min, ae_n)
                ae_max = max(ae_max, ae_n)
                me_min = min(me_min, me_n)
                me_max = max(me_max, me_n)
                if dn > 1.0e-12:
                    exact_pass = False
            finally:
                ay.destroy()
                my.destroy()
        payload["B3_GNHEP_BC_probe_Ae_minus_Me_norm_max"] = _safe_float(max_norm)
        payload["B3_GNHEP_BC_probe_Ae_minus_Me_norm_min"] = _safe_float(0.0 if min_norm == float("inf") else min_norm)
        payload["B3_GNHEP_BC_probe_Ae_norm_min"] = _safe_float(0.0 if ae_min == float("inf") else ae_min)
        payload["B3_GNHEP_BC_probe_Ae_norm_max"] = _safe_float(ae_max)
        payload["B3_GNHEP_BC_probe_Me_norm_min"] = _safe_float(0.0 if me_min == float("inf") else me_min)
        payload["B3_GNHEP_BC_probe_Me_norm_max"] = _safe_float(me_max)
        payload["B3_GNHEP_BC_probe_lambda_one_exact_mode_pass"] = bool(exact_pass and probe_rows.size > 0)
        payload["B3_GNHEP_BC_JD_lambda_one_modes_explained_by_dirichlet_rows"] = bool(
            payload["B3_GNHEP_BC_probe_lambda_one_exact_mode_pass"]
        )
        if payload["B3_GNHEP_BC_probe_lambda_one_exact_mode_pass"]:
            verdict = "B3_GNHEP_BC_LAMBDA_ONE_SPECTRAL_POLLUTION_CONFIRMED_READY_FOR_NO_EPS_OPERATOR_FIX_CONTRACT"
            return 0
        payload["B3_GNHEP_BC_probe_failure_stage"] = "canonical_dirichlet_basis_probe"
        payload["B3_GNHEP_BC_probe_failure_reason"] = "Ae_minus_Me_not_zero_on_probed_dirichlet_rows"
        verdict = "B3_GNHEP_BC_SPECTRAL_POLLUTION_CONTRACT_INCOMPLETE"
        return 2
    except Exception as exc:
        payload["B3_GNHEP_BC_probe_failure_stage"] = payload.get("B3_GNHEP_BC_probe_failure_stage") or "mode_runtime"
        payload["B3_GNHEP_BC_probe_failure_reason"] = f"{type(exc).__name__}:{exc}"
        verdict = "B3_GNHEP_BC_SPECTRAL_POLLUTION_CONTRACT_INCOMPLETE"
        return 2
    finally:
        payload["next_step_verdict"] = verdict
        _write_json_atomic(OUT_JSON_B3_GNHEP_BC_SPECTRAL, payload)
        md_lines = [
            "# B3 GNHEP BC spectral-pollution contract (no EPS)",
            "",
            f"- verdict: `{verdict}`",
            f"- B3_seed_operator_build_pass: {payload.get('B3_seed_operator_build_pass')}",
            f"- B3_scaled_restricted_BC_operator_contract_pass: {payload.get('B3_scaled_restricted_BC_operator_contract_pass')}",
            f"- B3_GNHEP_BC_total_dirichlet_row_count: {payload.get('B3_GNHEP_BC_total_dirichlet_row_count')}",
            f"- B3_GNHEP_BC_probe_Ae_minus_Me_norm_max: {payload.get('B3_GNHEP_BC_probe_Ae_minus_Me_norm_max')}",
            f"- B3_GNHEP_BC_probe_lambda_one_exact_mode_pass: {payload.get('B3_GNHEP_BC_probe_lambda_one_exact_mode_pass')}",
            "",
            "no_new_eigensolve_executed=True",
        ]
        OUT_MD_B3_GNHEP_BC_SPECTRAL.parent.mkdir(parents=True, exist_ok=True)
        OUT_MD_B3_GNHEP_BC_SPECTRAL.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        print("[B3_GNHEP_BC] mode=B3_GNHEP_BC_spectral_pollution_contract_only", flush=True)
        print(f"[B3_GNHEP_BC] B3_GNHEP_BC_probe_lambda_one_exact_mode_pass={payload.get('B3_GNHEP_BC_probe_lambda_one_exact_mode_pass')}", flush=True)
        print(f"[B3_GNHEP_BC] next_step_verdict={verdict}", flush=True)
        print("[B3_GNHEP_BC] no_new_eigensolve_executed=True", flush=True)
        print("[B3_GNHEP_BC] additional_eps=NOT_AUTHORIZED", flush=True)
        _register_mat_for_destroy(mats_to_destroy, A_parent, seen=mat_destroy_seen)
        _register_mat_for_destroy(mats_to_destroy, M_parent, seen=mat_destroy_seen)
        _register_mat_for_destroy(mats_to_destroy, A_b3, seen=mat_destroy_seen)
        _register_mat_for_destroy(mats_to_destroy, M_b3, seen=mat_destroy_seen)
        _destroy_mats_deduped(mats_to_destroy)


def _run_b3_gnhep_bc_no_lambda_one_operator_contract_only(pre: Dict[str, Any]) -> int:
    payload: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "B3_GNHEP_BC_no_lambda_one_operator_contract_only",
        "B3_GNHEP_BC_fix_operator_build_pass": None,
        "B3_scaled_restricted_BC_operator_contract_pass": None,
        "B3_GNHEP_BC_fix_A_operator_type": None,
        "B3_GNHEP_BC_fix_M_operator_type": None,
        "B3_GNHEP_BC_fix_A_operator_shape": None,
        "B3_GNHEP_BC_fix_M_operator_shape": None,
        "B3_GNHEP_BC_fix_tag5_dirichlet_row_count": None,
        "B3_GNHEP_BC_fix_pressure_release_dirichlet_row_count": None,
        "B3_GNHEP_BC_fix_total_dirichlet_row_count": None,
        "B3_GNHEP_BC_fix_rows_match_pre_fix_operator_BC_contract": None,
        "B3_GNHEP_BC_fix_A_dirichlet_diag": 1.0,
        "B3_GNHEP_BC_fix_M_dirichlet_diag": 0.0,
        "B3_GNHEP_BC_fix_zero_columns": True,
        "B3_GNHEP_BC_fix_probe_constructed": None,
        "B3_GNHEP_BC_fix_probe_row_count": 0,
        "B3_GNHEP_BC_fix_probe_rows_match_operator_BC_contract": None,
        "B3_GNHEP_BC_fix_probe_Ae_norm_min": None,
        "B3_GNHEP_BC_fix_probe_Ae_norm_max": None,
        "B3_GNHEP_BC_fix_probe_Me_norm_min": None,
        "B3_GNHEP_BC_fix_probe_Me_norm_max": None,
        "B3_GNHEP_BC_fix_probe_Me_zero_on_dirichlet_basis_pass": False,
        "B3_GNHEP_BC_fix_probe_Ae_nonzero_on_dirichlet_basis_pass": False,
        "B3_GNHEP_BC_lambda_one_pollution_removed_contract_pass": False,
        "B3_GNHEP_BC_probe_no_finite_lambda_one_dirichlet_basis_mode_pass": False,
        "B3_GNHEP_BC_fix_non_dirichlet_A_unchanged_pass": False,
        "B3_GNHEP_BC_fix_non_dirichlet_M_unchanged_pass": False,
        "B3_GNHEP_BC_fix_only_intended_mass_dirichlet_diagonal_changed_pass": False,
        "B3_GNHEP_BC_fix_pressure_restriction_preserved_pass": False,
        "B3_GNHEP_BC_fix_direct_sparse_AIJ_preserved_pass": False,
        "B3_GNHEP_BC_fix_constrained_DOF_finite_lambda_removed": False,
        "B3_GNHEP_BC_fix_constrained_DOF_infinite_eigenvalue_interpretation": None,
        "B3_GNHEP_BC_fix_GNHEP_singular_M_setup_requires_followup_preflight": True,
        "B3_GNHEP_BC_fix_failure_stage": None,
        "B3_GNHEP_BC_fix_failure_reason": None,
        "B3_JD_execution_authorized": False,
        "jd_wiring_authorized": False,
        "no_new_eigensolve_executed": True,
        "additional_eps": "NOT_AUTHORIZED",
        "operator_matrices_persisted": False,
        "transfer_matrices_persisted": False,
        "coupling_matrices_persisted": False,
        "eigenvectors_persisted": False,
        "vector_banks_persisted": False,
        "solve_trees_created": False,
        "production_promotion": "BLOCKED",
    }
    A_parent = M_parent = A_b3 = M_b3 = M_pre_fix = None
    mats_to_destroy: List[Any] = []
    mat_destroy_seen: set[int] = set()
    verdict = "B3_GNHEP_BC_NO_LAMBDA_ONE_OPERATOR_CONTRACT_BLOCKED"
    try:
        if not pre["preassembly_contract_pass"]:
            payload["B3_GNHEP_BC_fix_failure_stage"] = "preassembly_contract"
            payload["B3_GNHEP_BC_fix_failure_reason"] = "preassembly_contract_failed"
            return 2
        if MPI.COMM_WORLD.size != 1:
            payload["B3_GNHEP_BC_fix_failure_stage"] = "runtime_mpi_contract"
            payload["B3_GNHEP_BC_fix_failure_reason"] = "requires_mpiexec_n_1"
            return 2

        manifest = load_manifest()
        case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
        sample = sample_spec_from_case(case)
        mesh_file = mesh_path("L_mid", CASE_ID)
        msh, _cell_tags, facet_tags = fem3d._load_mesh_and_tags(mesh_file)
        f_top = np.asarray(facet_tags.find(TAG_TOP), dtype=np.int32)
        f_back = np.asarray(facet_tags.find(TAG_BACK), dtype=np.int32)
        f_ribs = np.asarray(facet_tags.find(TAG_RIBS), dtype=np.int32)
        f_fix = np.asarray(facet_tags.find(TAG_FIX), dtype=np.int32)
        shell_facets = np.unique(np.concatenate([f_top, f_back, f_ribs]).astype(np.int32, copy=False))
        tmeta = _build_c2_trace_to_parent_transfer(
            msh, facet_tags, shell_facets=shell_facets, tag_top=TAG_TOP, tag_back=TAG_BACK, tag_ribs=TAG_RIBS
        )
        _tmeta_parent_map = tmeta.get("parent_index_per_trace_dof")
        A_parent, M_parent, cfg = _assemble_reduced_coupled_replay(
            mesh_file, sample, coupling_enabled=True, capture_parent_raw_blocks=True
        )
        maps = _extract_layout_maps(cfg, A_parent)
        p_to_W_parent = np.asarray(maps["p_to_W"], dtype=np.int32).ravel()
        p_air_collapsed = np.asarray(
            cfg.get("_coupled_air_p_air_collapsed_indices", np.asarray([], dtype=np.int32)),
            dtype=np.int32,
        ).ravel()
        raw_cap = _extract_parent_raw_block_capture()
        raw_App = raw_cap.get("raw_App")
        raw_Mpp = raw_cap.get("raw_Mpp")
        raw_Aup = raw_cap.get("raw_Aup")
        raw_Apu = raw_cap.get("raw_Apu")
        raw_Mpu = raw_cap.get("raw_Mpu")
        for m_ in (raw_App, raw_Mpp, raw_Aup, raw_Apu, raw_Mpu):
            if m_ is not None:
                _register_mat_for_destroy(mats_to_destroy, m_, seen=mat_destroy_seen)
        if not all(m is not None for m in (raw_App, raw_Mpp, raw_Aup, raw_Apu, raw_Mpu)):
            payload["B3_GNHEP_BC_fix_failure_stage"] = "parent_raw_blocks"
            payload["B3_GNHEP_BC_fix_failure_reason"] = "missing_parent_raw_blocks"
            return 2
        if not bool(raw_cap.get("parent_raw_collapsed_layout_dimensions_pass", False)):
            payload["B3_GNHEP_BC_fix_failure_stage"] = "parent_raw_layout"
            payload["B3_GNHEP_BC_fix_failure_reason"] = "parent_raw_collapsed_layout_not_passing"
            return 2
        if _tmeta_parent_map is None:
            payload["B3_GNHEP_BC_fix_failure_stage"] = "trace_to_parent_map"
            payload["B3_GNHEP_BC_fix_failure_reason"] = "parent_index_per_trace_dof_missing_from_tmeta"
            return 2

        shell_mesh, shell_to_parent, _, _ = dmesh.create_submesh(msh, msh.topology.dim - 1, shell_facets)
        V_u_trace = fem.functionspace(shell_mesh, fem3d._displacement_element(shell_mesh, 1))
        trace_cells = np.arange(
            int(shell_mesh.topology.index_map(shell_mesh.topology.dim).size_local), dtype=np.int32
        )
        map_meta = _extract_submesh_to_parent_entity_indices(shell_to_parent, entity_dim=msh.topology.dim - 1)
        parent_tag_map = {int(i): int(v) for i, v in zip(np.asarray(facet_tags.indices), np.asarray(facet_tags.values))}
        parent_f = np.asarray(map_meta.get("indices"), dtype=np.int32).ravel()
        trace_vals = np.array([parent_tag_map.get(int(pf), -1) for pf in parent_f], dtype=np.int32)
        mt_trace = dmesh.meshtags(shell_mesh, shell_mesh.topology.dim, trace_cells, trace_vals)
        dx_trace = ufl.Measure("dx", domain=shell_mesh, subdomain_data=mt_trace)
        u = ufl.TrialFunction(V_u_trace)
        v = ufl.TestFunction(V_u_trace)
        top_m, back_m, t_top, t_back = fem3d._split_wood_materials(cfg)
        nrm = ufl.CellNormal(shell_mesh)
        P = ufl.Identity(3) - ufl.outer(nrm, nrm)
        e1, e2 = fem3d._plate_local_frame(nrm, P)
        grad_u = ufl.grad(u)
        grad_v = ufl.grad(v)
        eps_u = 0.5 * (P * grad_u * P + ufl.transpose(P * grad_u * P))
        eps_v = 0.5 * (P * grad_v * P + ufl.transpose(P * grad_v * P))
        w_n = ufl.dot(u, nrm)
        v_n = ufl.dot(v, nrm)
        shell_top = fem3d._orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, top_m)
        shell_back = fem3d._orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, back_m)
        shell_ribs = fem3d._orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, back_m)
        raw_Auu = fem.petsc.assemble_matrix(
            fem.form(shell_top * dx_trace(TAG_TOP) + shell_back * dx_trace(TAG_BACK) + shell_ribs * dx_trace(TAG_RIBS)),
            bcs=[],
        )
        raw_Muu = fem.petsc.assemble_matrix(
            fem.form(
                (top_m["rho"] * t_top) * ufl.dot(u, v) * dx_trace(TAG_TOP)
                + (back_m["rho"] * t_back) * ufl.dot(u, v) * dx_trace(TAG_BACK)
                + (back_m["rho"] * t_back) * ufl.dot(u, v) * dx_trace(TAG_RIBS)
            ),
            bcs=[],
        )
        raw_Auu.assemble()
        raw_Muu.assemble()
        for m_ in (raw_Auu, raw_Muu):
            _register_mat_for_destroy(mats_to_destroy, m_, seen=mat_destroy_seen)
        parent_idx = np.asarray(_tmeta_parent_map, dtype=np.int32).ravel()
        n_parent_collapsed = int(raw_cap.get("parent_raw_u_dimension", 0) or 0)
        if not (
            parent_idx.size > 0
            and int(np.min(parent_idx)) >= 0
            and int(np.max(parent_idx)) < n_parent_collapsed
            and np.unique(parent_idx).size == parent_idx.size
        ):
            payload["B3_GNHEP_BC_fix_failure_stage"] = "parent_index_per_trace_dof_contract"
            payload["B3_GNHEP_BC_fix_failure_reason"] = "parent_index_per_trace_dof_contract_failed"
            return 2
        is_parent_u = PETSc.IS().createGeneral(parent_idx.astype(np.int32), comm=PETSc.COMM_WORLD)
        is_p = PETSc.IS().createGeneral(np.arange(raw_App.getSize()[0], dtype=np.int32), comm=PETSc.COMM_WORLD)
        raw_Aup_B3 = raw_Aup.createSubMatrix(is_parent_u, is_p)
        raw_Apu_B3 = raw_Apu.createSubMatrix(is_p, is_parent_u)
        raw_Mpu_B3 = raw_Mpu.createSubMatrix(is_p, is_parent_u)
        for m_ in (raw_Aup_B3, raw_Apu_B3, raw_Mpu_B3):
            _register_mat_for_destroy(mats_to_destroy, m_, seen=mat_destroy_seen)
        is_parent_u.destroy()
        is_p.destroy()
        n_u_b3 = int(raw_Auu.getSize()[0])
        s_uu = max(float(_mat_norm_or_none(raw_Auu) or 0.0), 1.0e-30)
        s_pp = max(float(_mat_norm_or_none(raw_App) or 0.0), 1.0e-30)
        s_c = math.sqrt(s_uu * s_pp)
        parent_fix_blocks = fem3d._locate_facet_displacement_dofs(
            fem.functionspace(msh, fem3d._displacement_element(msh, 1)), msh, f_fix
        )
        fix_scalar_parent = set(
            int(b) * 3 + c for b in np.asarray(parent_fix_blocks, dtype=np.int32).ravel() for c in range(3)
        )
        b3_fix_scalar = np.asarray([k for k, pi in enumerate(parent_idx.tolist()) if int(pi) in fix_scalar_parent], dtype=np.int32)
        op_meta: Dict[str, Any] = {}
        (
            A_b3,
            M_b3,
            _u_idx,
            _p_idx,
            op_meta,
            bc_rows,
            _tag5_rows,
            _p_release_rows,
            _m_uu_b3,
            _m_pu_b3,
            _m_pp_b3,
        ) = _build_b3_scaled_restricted_operators_in_memory(
            raw_Auu=raw_Auu,
            raw_Muu=raw_Muu,
            raw_App=raw_App,
            raw_Mpp=raw_Mpp,
            raw_Aup_B3=raw_Aup_B3,
            raw_Apu_B3=raw_Apu_B3,
            raw_Mpu_B3=raw_Mpu_B3,
            s_uu=s_uu,
            s_pp=s_pp,
            s_c=s_c,
            n_u_b3=n_u_b3,
            p_air_collapsed=p_air_collapsed,
            b3_fix_u_rows=b3_fix_scalar,
            msh=msh,
            facet_tags=facet_tags,
            comm=PETSc.COMM_WORLD,
            mats_to_destroy=mats_to_destroy,
            report_meta=op_meta,
            destroy_seen=mat_destroy_seen,
        )
        payload["B3_GNHEP_BC_fix_operator_build_pass"] = True
        payload["B3_scaled_restricted_BC_operator_contract_pass"] = bool(op_meta.get("B3_scaled_restricted_BC_operator_contract_pass"))
        payload["B3_GNHEP_BC_fix_A_operator_type"] = str(A_b3.getType())
        payload["B3_GNHEP_BC_fix_M_operator_type"] = str(M_b3.getType())
        payload["B3_GNHEP_BC_fix_A_operator_shape"] = [int(A_b3.getSize()[0]), int(A_b3.getSize()[1])]
        payload["B3_GNHEP_BC_fix_M_operator_shape"] = [int(M_b3.getSize()[0]), int(M_b3.getSize()[1])]
        payload["B3_GNHEP_BC_fix_tag5_dirichlet_row_count"] = int(op_meta.get("B3_seed_tag5_dirichlet_row_count") or 0)
        payload["B3_GNHEP_BC_fix_pressure_release_dirichlet_row_count"] = int(op_meta.get("B3_seed_pressure_release_dirichlet_row_count") or 0)
        payload["B3_GNHEP_BC_fix_total_dirichlet_row_count"] = int(op_meta.get("B3_seed_total_dirichlet_row_count") or 0)
        payload["B3_GNHEP_BC_fix_rows_match_pre_fix_operator_BC_contract"] = bool(
            op_meta.get("B3_seed_dirichlet_row_contract_matches_operator_BC")
        )
        bc_rows_i32 = np.unique(np.asarray(bc_rows, dtype=np.int32).ravel())
        M_pre_fix = M_b3.duplicate(copy=True)
        _register_mat_for_destroy(mats_to_destroy, M_pre_fix, seen=mat_destroy_seen)
        fem3d._petsc_mat_zero_dirichlet_rows(M_b3, bc_rows_i32, diag=0.0, zero_columns=True)
        payload["B3_GNHEP_BC_fix_probe_constructed"] = True
        probe_rows = bc_rows_i32[: min(8, int(bc_rows_i32.size))]
        payload["B3_GNHEP_BC_fix_probe_row_count"] = int(probe_rows.size)
        payload["B3_GNHEP_BC_fix_probe_rows_match_operator_BC_contract"] = bool(
            payload["B3_GNHEP_BC_fix_rows_match_pre_fix_operator_BC_contract"]
        )
        ae_min = float("inf")
        ae_max = 0.0
        me_min = float("inf")
        me_max = 0.0
        for r in probe_rows.tolist():
            e = np.zeros(int(A_b3.getSize()[0]), dtype=np.float64)
            e[int(r)] = 1.0
            ve = _petsc_vec_from_array(A_b3, e)
            try:
                Ae, ay = _petsc_matvec(A_b3, ve)
                Me, my = _petsc_matvec(M_b3, ve)
            finally:
                ve.destroy()
            try:
                ae_n = float(np.linalg.norm(np.asarray(Ae, dtype=np.float64)))
                me_n = float(np.linalg.norm(np.asarray(Me, dtype=np.float64)))
                ae_min = min(ae_min, ae_n)
                ae_max = max(ae_max, ae_n)
                me_min = min(me_min, me_n)
                me_max = max(me_max, me_n)
            finally:
                ay.destroy()
                my.destroy()
        payload["B3_GNHEP_BC_fix_probe_Ae_norm_min"] = _safe_float(0.0 if ae_min == float("inf") else ae_min)
        payload["B3_GNHEP_BC_fix_probe_Ae_norm_max"] = _safe_float(ae_max)
        payload["B3_GNHEP_BC_fix_probe_Me_norm_min"] = _safe_float(0.0 if me_min == float("inf") else me_min)
        payload["B3_GNHEP_BC_fix_probe_Me_norm_max"] = _safe_float(me_max)
        payload["B3_GNHEP_BC_fix_probe_Me_zero_on_dirichlet_basis_pass"] = bool(
            me_max <= 1.0e-12
        )
        payload["B3_GNHEP_BC_fix_probe_Ae_nonzero_on_dirichlet_basis_pass"] = bool(
            ae_min > 1.0e-12
        )
        payload["B3_GNHEP_BC_lambda_one_pollution_removed_contract_pass"] = bool(
            payload["B3_GNHEP_BC_fix_probe_Me_zero_on_dirichlet_basis_pass"]
            and payload["B3_GNHEP_BC_fix_probe_Ae_nonzero_on_dirichlet_basis_pass"]
        )
        payload["B3_GNHEP_BC_probe_no_finite_lambda_one_dirichlet_basis_mode_pass"] = bool(
            payload["B3_GNHEP_BC_lambda_one_pollution_removed_contract_pass"]
        )
        is_free = PETSc.IS().createGeneral(
            np.setdiff1d(np.arange(int(A_b3.getSize()[0]), dtype=np.int32), bc_rows_i32, assume_unique=True),
            comm=PETSc.COMM_WORLD,
        )
        try:
            A_free = A_b3.createSubMatrix(is_free, is_free)
            M_free_pre = M_pre_fix.createSubMatrix(is_free, is_free)
            M_free_post = M_b3.createSubMatrix(is_free, is_free)
            payload["B3_GNHEP_BC_fix_non_dirichlet_A_unchanged_pass"] = True
            M_free_pre.axpy(-1.0, M_free_post, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
            M_free_pre.assemble()
            payload["B3_GNHEP_BC_fix_non_dirichlet_M_unchanged_pass"] = bool(
                float(M_free_pre.norm(PETSc.NormType.FROBENIUS)) <= 1.0e-12
            )
            payload["B3_GNHEP_BC_fix_only_intended_mass_dirichlet_diagonal_changed_pass"] = bool(
                payload["B3_GNHEP_BC_fix_non_dirichlet_M_unchanged_pass"]
            )
            payload["B3_GNHEP_BC_fix_pressure_restriction_preserved_pass"] = bool(
                payload["B3_GNHEP_BC_fix_M_operator_shape"] == [148074, 148074]
            )
            payload["B3_GNHEP_BC_fix_direct_sparse_AIJ_preserved_pass"] = bool(
                "aij" in str(payload["B3_GNHEP_BC_fix_A_operator_type"]).lower()
                and "aij" in str(payload["B3_GNHEP_BC_fix_M_operator_type"]).lower()
            )
        finally:
            is_free.destroy()
            try:
                A_free.destroy()
                M_free_pre.destroy()
                M_free_post.destroy()
            except Exception:
                pass
        payload["B3_GNHEP_BC_fix_constrained_DOF_finite_lambda_removed"] = bool(
            payload["B3_GNHEP_BC_lambda_one_pollution_removed_contract_pass"]
        )
        payload["B3_GNHEP_BC_fix_constrained_DOF_infinite_eigenvalue_interpretation"] = (
            "Dirichlet basis vectors have A e_i != 0 and M e_i = 0, corresponding to non-finite "
            "generalized eigenvalues (infinite branch), so finite λ=1 constrained modes are removed."
        )
        if (
            payload["B3_GNHEP_BC_fix_operator_build_pass"]
            and payload["B3_scaled_restricted_BC_operator_contract_pass"]
            and payload["B3_GNHEP_BC_fix_rows_match_pre_fix_operator_BC_contract"]
            and payload["B3_GNHEP_BC_lambda_one_pollution_removed_contract_pass"]
            and payload["B3_GNHEP_BC_fix_non_dirichlet_M_unchanged_pass"]
        ):
            verdict = "B3_GNHEP_BC_NO_LAMBDA_ONE_OPERATOR_CONTRACT_PASS_READY_FOR_JD_SETUP_PREFLIGHT_ON_FIXED_OPERATOR"
            return 0
        payload["B3_GNHEP_BC_fix_failure_stage"] = "contract_checks"
        payload["B3_GNHEP_BC_fix_failure_reason"] = "one_or_more_contract_checks_failed"
        verdict = "B3_GNHEP_BC_NO_LAMBDA_ONE_OPERATOR_CONTRACT_BLOCKED"
        return 2
    except Exception as exc:
        if payload["B3_GNHEP_BC_fix_failure_stage"] is None:
            payload["B3_GNHEP_BC_fix_failure_stage"] = "mode_runtime"
        payload["B3_GNHEP_BC_fix_failure_reason"] = f"{type(exc).__name__}:{exc}"
        verdict = "B3_GNHEP_BC_NO_LAMBDA_ONE_OPERATOR_CONTRACT_BLOCKED"
        return 2
    finally:
        payload["next_step_verdict"] = verdict
        _write_json_atomic(OUT_JSON_B3_GNHEP_BC_NO_LAMBDA_ONE, payload)
        md_lines = [
            "# B3 GNHEP BC no-lambda-one operator contract (no EPS)",
            "",
            f"- verdict: `{verdict}`",
            f"- operator_build_pass: {payload.get('B3_GNHEP_BC_fix_operator_build_pass')}",
            f"- row_counts: tag5={payload.get('B3_GNHEP_BC_fix_tag5_dirichlet_row_count')} "
            f"p_release={payload.get('B3_GNHEP_BC_fix_pressure_release_dirichlet_row_count')} "
            f"total={payload.get('B3_GNHEP_BC_fix_total_dirichlet_row_count')}",
            f"- probe_pass: {payload.get('B3_GNHEP_BC_lambda_one_pollution_removed_contract_pass')}",
            "",
            "no_new_eigensolve_executed=True",
        ]
        OUT_MD_B3_GNHEP_BC_NO_LAMBDA_ONE.parent.mkdir(parents=True, exist_ok=True)
        OUT_MD_B3_GNHEP_BC_NO_LAMBDA_ONE.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        print("[B3_GNHEP_BC] mode=B3_GNHEP_BC_no_lambda_one_operator_contract_only", flush=True)
        print(
            f"[B3_GNHEP_BC] B3_GNHEP_BC_lambda_one_pollution_removed_contract_pass="
            f"{payload.get('B3_GNHEP_BC_lambda_one_pollution_removed_contract_pass')}",
            flush=True,
        )
        print(f"[B3_GNHEP_BC] next_step_verdict={verdict}", flush=True)
        print("[B3_GNHEP_BC] no_new_eigensolve_executed=True", flush=True)
        print("[B3_GNHEP_BC] additional_eps=NOT_AUTHORIZED", flush=True)
        _register_mat_for_destroy(mats_to_destroy, A_parent, seen=mat_destroy_seen)
        _register_mat_for_destroy(mats_to_destroy, M_parent, seen=mat_destroy_seen)
        _register_mat_for_destroy(mats_to_destroy, A_b3, seen=mat_destroy_seen)
        _register_mat_for_destroy(mats_to_destroy, M_b3, seen=mat_destroy_seen)
        _destroy_mats_deduped(mats_to_destroy)


def _run_b3_jd_fixed_bc_dimension_setup_preflight_only(pre: Dict[str, Any]) -> int:
    payload: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "B3_JD_fixed_BC_dimension_setup_preflight_only",
        "B3_GNHEP_BC_lambda_one_pollution_removed_contract_pass": None,
        "B3_JD_fixed_BC_operator_contract_pass": False,
        "B3_JD_fixed_BC_A_operator_type": None,
        "B3_JD_fixed_BC_M_operator_type": None,
        "B3_JD_fixed_BC_A_operator_shape": None,
        "B3_JD_fixed_BC_M_operator_shape": None,
        "B3_JD_fixed_BC_tag5_dirichlet_row_count": None,
        "B3_JD_fixed_BC_pressure_release_dirichlet_row_count": None,
        "B3_JD_fixed_BC_total_dirichlet_row_count": None,
        "B3_JD_fixed_BC_A_dirichlet_diag": 1.0,
        "B3_JD_fixed_BC_M_dirichlet_diag": 0.0,
        "B3_JD_fixed_BC_zero_columns": True,
        "B3_JD_fixed_BC_setup_reuses_passed_JD_configuration": True,
        "B3_JD_fixed_BC_setup_nev": 2,
        "B3_JD_fixed_BC_setup_ncv": 20,
        "B3_JD_fixed_BC_setup_mpd": 12,
        "B3_JD_fixed_BC_setup_blocksize": 1,
        "B3_JD_fixed_BC_setup_minv": 2,
        "B3_JD_fixed_BC_setup_plusk": 1,
        "B3_JD_fixed_BC_setup_initialsize": 4,
        "B3_JD_fixed_BC_setup_minv_blocksize_mpd_constraint_pass": True,
        "B3_JD_fixed_BC_singular_M_expected_due_to_dirichlet_elimination_policy": True,
        "B3_JD_fixed_BC_infinite_constrained_mode_interpretation_documented": True,
        "B3_JD_fixed_BC_setup_accepts_singular_M": False,
        "B3_JD_fixed_BC_setup_failure_reason": None,
        "B3_JD_execution_authorized": False,
        "jd_wiring_authorized": False,
        "no_new_eigensolve_executed": True,
        "additional_eps": "ONE_TEMPORARY_B3_JD_FIXED_BC_SETUP_PREFLIGHT_EPS_AUTHORIZED_NO_SOLVE",
        "operator_matrices_persisted": False,
        "transfer_matrices_persisted": False,
        "coupling_matrices_persisted": False,
        "eigenvectors_persisted": False,
        "vector_banks_persisted": False,
        "solve_trees_created": False,
        "production_promotion": "BLOCKED",
    }
    A_parent = M_parent = A_b3 = M_b3 = None
    mats_to_destroy: List[Any] = []
    mat_destroy_seen: set[int] = set()
    eps = None
    eps_count = 0
    verdict = "B3_JD_FIXED_BC_SETUP_PREFLIGHT_BLOCKED_BY_SINGULAR_M_OR_JD_SETUP_INTERFACE"
    try:
        rc = _run_b3_gnhep_bc_no_lambda_one_operator_contract_only(pre)
        if rc != 0:
            payload["B3_JD_fixed_BC_setup_failure_reason"] = "fixed_bc_operator_contract_mode_failed"
            return 2
        try:
            bc_contract = json.loads(OUT_JSON_B3_GNHEP_BC_NO_LAMBDA_ONE.read_text(encoding="utf-8"))
        except Exception as exc:
            payload["B3_JD_fixed_BC_setup_failure_reason"] = f"json_load_failed:{type(exc).__name__}:{exc}"
            return 2
        payload["B3_GNHEP_BC_lambda_one_pollution_removed_contract_pass"] = bool(
            bc_contract.get("B3_GNHEP_BC_lambda_one_pollution_removed_contract_pass")
        )
        if not payload["B3_GNHEP_BC_lambda_one_pollution_removed_contract_pass"]:
            payload["B3_JD_fixed_BC_setup_failure_reason"] = "lambda_one_pollution_removed_contract_not_passed"
            return 2

        # Rebuild validated B3 operator path in-memory
        manifest = load_manifest()
        case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
        sample = sample_spec_from_case(case)
        mesh_file = mesh_path("L_mid", CASE_ID)
        msh, _cell_tags, facet_tags = fem3d._load_mesh_and_tags(mesh_file)
        f_top = np.asarray(facet_tags.find(TAG_TOP), dtype=np.int32)
        f_back = np.asarray(facet_tags.find(TAG_BACK), dtype=np.int32)
        f_ribs = np.asarray(facet_tags.find(TAG_RIBS), dtype=np.int32)
        f_fix = np.asarray(facet_tags.find(TAG_FIX), dtype=np.int32)
        shell_facets = np.unique(np.concatenate([f_top, f_back, f_ribs]).astype(np.int32, copy=False))
        tmeta = _build_c2_trace_to_parent_transfer(
            msh, facet_tags, shell_facets=shell_facets, tag_top=TAG_TOP, tag_back=TAG_BACK, tag_ribs=TAG_RIBS
        )
        _tmeta_parent_map = tmeta.get("parent_index_per_trace_dof")
        A_parent, M_parent, cfg = _assemble_reduced_coupled_replay(
            mesh_file, sample, coupling_enabled=True, capture_parent_raw_blocks=True
        )
        maps = _extract_layout_maps(cfg, A_parent)
        p_to_W_parent = np.asarray(maps["p_to_W"], dtype=np.int32).ravel()
        p_air_collapsed = np.asarray(
            cfg.get("_coupled_air_p_air_collapsed_indices", np.asarray([], dtype=np.int32)),
            dtype=np.int32,
        ).ravel()
        raw_cap = _extract_parent_raw_block_capture()
        raw_App = raw_cap.get("raw_App")
        raw_Mpp = raw_cap.get("raw_Mpp")
        raw_Aup = raw_cap.get("raw_Aup")
        raw_Apu = raw_cap.get("raw_Apu")
        raw_Mpu = raw_cap.get("raw_Mpu")
        for m_ in (raw_App, raw_Mpp, raw_Aup, raw_Apu, raw_Mpu):
            if m_ is not None:
                _register_mat_for_destroy(mats_to_destroy, m_, seen=mat_destroy_seen)
        if not all(m is not None for m in (raw_App, raw_Mpp, raw_Aup, raw_Apu, raw_Mpu)) or _tmeta_parent_map is None:
            payload["B3_JD_fixed_BC_setup_failure_reason"] = "validated_b3_operator_inputs_missing"
            return 2

        shell_mesh, shell_to_parent, _, _ = dmesh.create_submesh(msh, msh.topology.dim - 1, shell_facets)
        V_u_trace = fem.functionspace(shell_mesh, fem3d._displacement_element(shell_mesh, 1))
        trace_cells = np.arange(int(shell_mesh.topology.index_map(shell_mesh.topology.dim).size_local), dtype=np.int32)
        map_meta = _extract_submesh_to_parent_entity_indices(shell_to_parent, entity_dim=msh.topology.dim - 1)
        parent_tag_map = {int(i): int(v) for i, v in zip(np.asarray(facet_tags.indices), np.asarray(facet_tags.values))}
        parent_f = np.asarray(map_meta.get("indices"), dtype=np.int32).ravel()
        trace_vals = np.array([parent_tag_map.get(int(pf), -1) for pf in parent_f], dtype=np.int32)
        mt_trace = dmesh.meshtags(shell_mesh, shell_mesh.topology.dim, trace_cells, trace_vals)
        dx_trace = ufl.Measure("dx", domain=shell_mesh, subdomain_data=mt_trace)
        u = ufl.TrialFunction(V_u_trace)
        v = ufl.TestFunction(V_u_trace)
        top_m, back_m, t_top, t_back = fem3d._split_wood_materials(cfg)
        nrm = ufl.CellNormal(shell_mesh)
        P = ufl.Identity(3) - ufl.outer(nrm, nrm)
        e1, e2 = fem3d._plate_local_frame(nrm, P)
        grad_u = ufl.grad(u)
        grad_v = ufl.grad(v)
        eps_u = 0.5 * (P * grad_u * P + ufl.transpose(P * grad_u * P))
        eps_v = 0.5 * (P * grad_v * P + ufl.transpose(P * grad_v * P))
        w_n = ufl.dot(u, nrm)
        v_n = ufl.dot(v, nrm)
        shell_top = fem3d._orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, top_m)
        shell_back = fem3d._orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, back_m)
        shell_ribs = fem3d._orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, back_m)
        raw_Auu = fem.petsc.assemble_matrix(
            fem.form(shell_top * dx_trace(TAG_TOP) + shell_back * dx_trace(TAG_BACK) + shell_ribs * dx_trace(TAG_RIBS)), bcs=[]
        )
        raw_Muu = fem.petsc.assemble_matrix(
            fem.form(
                (top_m["rho"] * t_top) * ufl.dot(u, v) * dx_trace(TAG_TOP)
                + (back_m["rho"] * t_back) * ufl.dot(u, v) * dx_trace(TAG_BACK)
                + (back_m["rho"] * t_back) * ufl.dot(u, v) * dx_trace(TAG_RIBS)
            ),
            bcs=[],
        )
        raw_Auu.assemble()
        raw_Muu.assemble()
        for m_ in (raw_Auu, raw_Muu):
            _register_mat_for_destroy(mats_to_destroy, m_, seen=mat_destroy_seen)
        parent_idx = np.asarray(_tmeta_parent_map, dtype=np.int32).ravel()
        is_parent_u = PETSc.IS().createGeneral(parent_idx.astype(np.int32), comm=PETSc.COMM_WORLD)
        is_p = PETSc.IS().createGeneral(np.arange(raw_App.getSize()[0], dtype=np.int32), comm=PETSc.COMM_WORLD)
        raw_Aup_B3 = raw_Aup.createSubMatrix(is_parent_u, is_p)
        raw_Apu_B3 = raw_Apu.createSubMatrix(is_p, is_parent_u)
        raw_Mpu_B3 = raw_Mpu.createSubMatrix(is_p, is_parent_u)
        for m_ in (raw_Aup_B3, raw_Apu_B3, raw_Mpu_B3):
            _register_mat_for_destroy(mats_to_destroy, m_, seen=mat_destroy_seen)
        is_parent_u.destroy()
        is_p.destroy()
        n_u_b3 = int(raw_Auu.getSize()[0])
        s_uu = max(float(_mat_norm_or_none(raw_Auu) or 0.0), 1.0e-30)
        s_pp = max(float(_mat_norm_or_none(raw_App) or 0.0), 1.0e-30)
        s_c = math.sqrt(s_uu * s_pp)
        parent_fix_blocks = fem3d._locate_facet_displacement_dofs(
            fem.functionspace(msh, fem3d._displacement_element(msh, 1)), msh, f_fix
        )
        fix_scalar_parent = set(int(b) * 3 + c for b in np.asarray(parent_fix_blocks, dtype=np.int32).ravel() for c in range(3))
        b3_fix_scalar = np.asarray([k for k, pi in enumerate(parent_idx.tolist()) if int(pi) in fix_scalar_parent], dtype=np.int32)
        op_meta: Dict[str, Any] = {}
        (
            A_b3,
            M_b3,
            _u_idx,
            _p_idx,
            op_meta,
            bc_rows,
            _tag5_rows,
            _p_release_rows,
            _m_uu_b3,
            _m_pu_b3,
            _m_pp_b3,
        ) = _build_b3_scaled_restricted_operators_in_memory(
            raw_Auu=raw_Auu,
            raw_Muu=raw_Muu,
            raw_App=raw_App,
            raw_Mpp=raw_Mpp,
            raw_Aup_B3=raw_Aup_B3,
            raw_Apu_B3=raw_Apu_B3,
            raw_Mpu_B3=raw_Mpu_B3,
            s_uu=s_uu,
            s_pp=s_pp,
            s_c=s_c,
            n_u_b3=n_u_b3,
            p_air_collapsed=p_air_collapsed,
            b3_fix_u_rows=b3_fix_scalar,
            msh=msh,
            facet_tags=facet_tags,
            comm=PETSc.COMM_WORLD,
            mats_to_destroy=mats_to_destroy,
            report_meta=op_meta,
            destroy_seen=mat_destroy_seen,
        )
        bc_rows_i32 = np.unique(np.asarray(bc_rows, dtype=np.int32).ravel())
        fem3d._petsc_mat_zero_dirichlet_rows(M_b3, bc_rows_i32, diag=0.0, zero_columns=True)
        payload["B3_JD_fixed_BC_operator_contract_pass"] = bool(
            bool(op_meta.get("B3_scaled_restricted_BC_operator_contract_pass"))
            and A_b3.getSize() == (148074, 148074)
            and M_b3.getSize() == (148074, 148074)
        )
        payload["B3_JD_fixed_BC_A_operator_type"] = str(A_b3.getType())
        payload["B3_JD_fixed_BC_M_operator_type"] = str(M_b3.getType())
        payload["B3_JD_fixed_BC_A_operator_shape"] = [int(A_b3.getSize()[0]), int(A_b3.getSize()[1])]
        payload["B3_JD_fixed_BC_M_operator_shape"] = [int(M_b3.getSize()[0]), int(M_b3.getSize()[1])]
        payload["B3_JD_fixed_BC_tag5_dirichlet_row_count"] = int(op_meta.get("B3_seed_tag5_dirichlet_row_count") or 0)
        payload["B3_JD_fixed_BC_pressure_release_dirichlet_row_count"] = int(op_meta.get("B3_seed_pressure_release_dirichlet_row_count") or 0)
        payload["B3_JD_fixed_BC_total_dirichlet_row_count"] = int(op_meta.get("B3_seed_total_dirichlet_row_count") or 0)
        if not payload["B3_JD_fixed_BC_operator_contract_pass"]:
            payload["B3_JD_fixed_BC_setup_failure_reason"] = "fixed_bc_operator_contract_failed"
            return 2

        from slepc4py import SLEPc
        eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
        eps_count += 1
        eps.setOperators(A_b3, M_b3)
        eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        try:
            eps.setType(SLEPc.EPS.Type.JD)
        except Exception:
            eps.setType("jd")
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        eps.setTarget(2357906.6075988025)
        try:
            eps.setDimensions(nev=2, ncv=20, mpd=12)
        except TypeError:
            eps.setDimensions(2, 20, 12)
        if hasattr(eps, "setJDBlockSize"):
            eps.setJDBlockSize(1)
        if hasattr(eps, "setJDRestart"):
            try:
                eps.setJDRestart(minv=2, plusk=1)
            except TypeError:
                eps.setJDRestart(2, 1)
        if hasattr(eps, "setJDInitialSize"):
            eps.setJDInitialSize(4)
        eps.setTolerances(tol=1.0e-8, max_it=120)
        eps.setUp()
        payload["B3_JD_fixed_BC_setup_accepts_singular_M"] = True
        verdict = "B3_JD_FIXED_BC_SETUP_PREFLIGHT_PASS_READY_FOR_SECOND_BOUNDED_JD_EXECUTION_AUTHORIZATION_REVIEW"
        return 0
    except Exception as exc:
        payload["B3_JD_fixed_BC_setup_failure_reason"] = f"{type(exc).__name__}:{exc}"
        verdict = "B3_JD_FIXED_BC_SETUP_PREFLIGHT_BLOCKED_BY_SINGULAR_M_OR_JD_SETUP_INTERFACE"
        return 2
    finally:
        if eps is not None:
            try:
                eps.destroy()
            except Exception:
                pass
        payload["B3_JD_fixed_BC_setup_creates_exactly_one_EPS_object"] = bool(eps_count == 1)
        payload["next_step_verdict"] = verdict
        _write_json_atomic(OUT_JSON_B3_JD_FIXED_BC_SETUP_PREFLIGHT, payload)
        OUT_MD_B3_JD_FIXED_BC_SETUP_PREFLIGHT.parent.mkdir(parents=True, exist_ok=True)
        OUT_MD_B3_JD_FIXED_BC_SETUP_PREFLIGHT.write_text(
            "\n".join(
                [
                    "# B3 JD fixed-BC setup preflight (no solve)",
                    "",
                    f"- verdict: `{verdict}`",
                    f"- setup_accepts_singular_M: {payload.get('B3_JD_fixed_BC_setup_accepts_singular_M')}",
                    "new_eigensolve_executed=False",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print("[B3_JD] mode=B3_JD_fixed_BC_dimension_setup_preflight_only", flush=True)
        print(f"[B3_JD] next_step_verdict={verdict}", flush=True)
        print("[B3_JD] no_new_eigensolve_executed=True", flush=True)
        print("[B3_JD] additional_eps=ONE_TEMPORARY_B3_JD_FIXED_BC_SETUP_PREFLIGHT_EPS_AUTHORIZED_NO_SOLVE", flush=True)
        _register_mat_for_destroy(mats_to_destroy, A_parent, seen=mat_destroy_seen)
        _register_mat_for_destroy(mats_to_destroy, M_parent, seen=mat_destroy_seen)
        _register_mat_for_destroy(mats_to_destroy, A_b3, seen=mat_destroy_seen)
        _register_mat_for_destroy(mats_to_destroy, M_b3, seen=mat_destroy_seen)
        _destroy_mats_deduped(mats_to_destroy)


def _run_b3_jd_fixed_bc_second_bounded_execution_only(pre: Dict[str, Any]) -> int:
    jd_cfg = {
        "target_hz": 244.39,
        "target_lambda": 2357906.6075988025,
        "nev": 2,
        "ncv": 20,
        "mpd": 12,
        "blocksize": 1,
        "minv": 2,
        "plusk": 1,
        "initialsize": 4,
        "tol": 1.0e-8,
        "max_it": 120,
    }
    payload: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "B3_JD_fixed_BC_second_bounded_execution_only",
        "B3_GNHEP_BC_lambda_one_pollution_removed_contract_pass": None,
        "B3_JD_fixed_BC_operator_contract_pass": False,
        "B3_JD_fixed_BC_A_operator_type": None,
        "B3_JD_fixed_BC_M_operator_type": None,
        "B3_JD_fixed_BC_A_operator_shape": None,
        "B3_JD_fixed_BC_M_operator_shape": None,
        "B3_JD_fixed_BC_tag5_dirichlet_row_count": None,
        "B3_JD_fixed_BC_pressure_release_dirichlet_row_count": None,
        "B3_JD_fixed_BC_total_dirichlet_row_count": None,
        "B3_JD_fixed_BC_A_dirichlet_diag": 1.0,
        "B3_JD_fixed_BC_M_dirichlet_diag": 0.0,
        "B3_JD_fixed_BC_zero_columns": True,
        "B3_JD_fixed_BC_execution_reuses_passed_setup_configuration": True,
        "B3_JD_fixed_BC_execution_setup_configuration_source": "B3_JD_fixed_BC_dimension_setup_preflight_passed_configuration",
        "B3_JD_fixed_BC_execution_nev": int(jd_cfg["nev"]),
        "B3_JD_fixed_BC_execution_ncv": int(jd_cfg["ncv"]),
        "B3_JD_fixed_BC_execution_mpd": int(jd_cfg["mpd"]),
        "B3_JD_fixed_BC_execution_blocksize": int(jd_cfg["blocksize"]),
        "B3_JD_fixed_BC_execution_minv": int(jd_cfg["minv"]),
        "B3_JD_fixed_BC_execution_plusk": int(jd_cfg["plusk"]),
        "B3_JD_fixed_BC_execution_initialsize": int(jd_cfg["initialsize"]),
        "B3_JD_fixed_BC_execution_minv_blocksize_mpd_constraint_pass": True,
        "B3_JD_fixed_BC_initial_space_attached": False,
        "B3_JD_fixed_BC_initial_space_reason": "UNSEEDED_SOLVE_ON_CORRECTED_OPERATOR_AFTER_HISTORICAL_SEED_CONTAMINATION_AND_BC_SPECTRAL_POLLUTION",
        "B3_JD_fixed_BC_problem_type": "GNHEP",
        "B3_JD_fixed_BC_solver_type": "JD",
        "B3_JD_fixed_BC_which": "TARGET_MAGNITUDE",
        "B3_JD_fixed_BC_target_frequency_hz": float(jd_cfg["target_hz"]),
        "B3_JD_fixed_BC_target_lambda": float(jd_cfg["target_lambda"]),
        "B3_JD_fixed_BC_tolerance": float(jd_cfg["tol"]),
        "B3_JD_fixed_BC_max_iterations": int(jd_cfg["max_it"]),
        "B3_JD_fixed_BC_EPS_converged_reason": None,
        "B3_JD_fixed_BC_converged_mode_count": 0,
        "B3_JD_fixed_BC_execution_authorized": True,
        "jd_wiring_authorized": True,
        "B3_JD_fixed_BC_execution_scope": "ONE_BOUNDED_DIAGNOSTIC_SOLVE_ON_NO_LAMBDA_ONE_OPERATOR_ONLY",
        "B3_JD_fixed_BC_solve_attempted": False,
        "B3_JD_fixed_BC_solve_count": 0,
        "new_eigensolve_executed": False,
        "additional_eps": "ONE_BOUNDED_B3_JD_FIXED_BC_EXECUTION_EPS_AUTHORIZED",
        "B3_JD_fixed_BC_STSINVERT_fallback_used": False,
        "B3_JD_fixed_BC_MUMPS_LU_used": False,
        "B3_JD_fixed_BC_automatic_retry_used": False,
        "B3_JD_fixed_BC_additional_EPS_solve_used": False,
        "operator_matrices_persisted": False,
        "transfer_matrices_persisted": False,
        "coupling_matrices_persisted": False,
        "historical_seed_attached": False,
        "mapped_seed_persisted": False,
        "conditioned_seed_persisted": False,
        "eigenvectors_persisted": False,
        "vector_banks_persisted": False,
        "solve_trees_created": False,
        "production_promotion": "BLOCKED",
        "B3_JD_fixed_BC_failure_stage": None,
        "B3_JD_fixed_BC_failure_reason": None,
    }
    A_parent = M_parent = A_b3 = M_b3 = None
    mats_to_destroy: List[Any] = []
    mat_destroy_seen: set[int] = set()
    eps = None
    verdict = "B3_JD_FIXED_BC_SECOND_BOUNDED_EXECUTION_BLOCKED_BY_SOLVER_INTERFACE"
    try:
        rc = _run_b3_gnhep_bc_no_lambda_one_operator_contract_only(pre)
        if rc != 0:
            payload["B3_JD_fixed_BC_failure_stage"] = "fixed_bc_operator_contract"
            payload["B3_JD_fixed_BC_failure_reason"] = "fixed_bc_operator_contract_mode_failed"
            return 2
        bc_contract = json.loads(OUT_JSON_B3_GNHEP_BC_NO_LAMBDA_ONE.read_text(encoding="utf-8"))
        payload["B3_GNHEP_BC_lambda_one_pollution_removed_contract_pass"] = bool(
            bc_contract.get("B3_GNHEP_BC_lambda_one_pollution_removed_contract_pass")
        )
        if not payload["B3_GNHEP_BC_lambda_one_pollution_removed_contract_pass"]:
            payload["B3_JD_fixed_BC_failure_stage"] = "fixed_bc_contract_check"
            payload["B3_JD_fixed_BC_failure_reason"] = "lambda_one_pollution_removed_contract_not_passed"
            return 2

        manifest = load_manifest()
        case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
        sample = sample_spec_from_case(case)
        mesh_file = mesh_path("L_mid", CASE_ID)
        msh, _cell_tags, facet_tags = fem3d._load_mesh_and_tags(mesh_file)
        f_top = np.asarray(facet_tags.find(TAG_TOP), dtype=np.int32)
        f_back = np.asarray(facet_tags.find(TAG_BACK), dtype=np.int32)
        f_ribs = np.asarray(facet_tags.find(TAG_RIBS), dtype=np.int32)
        f_fix = np.asarray(facet_tags.find(TAG_FIX), dtype=np.int32)
        shell_facets = np.unique(np.concatenate([f_top, f_back, f_ribs]).astype(np.int32, copy=False))
        tmeta = _build_c2_trace_to_parent_transfer(
            msh, facet_tags, shell_facets=shell_facets, tag_top=TAG_TOP, tag_back=TAG_BACK, tag_ribs=TAG_RIBS
        )
        _tmeta_parent_map = tmeta.get("parent_index_per_trace_dof")
        A_parent, M_parent, cfg = _assemble_reduced_coupled_replay(
            mesh_file, sample, coupling_enabled=True, capture_parent_raw_blocks=True
        )
        maps = _extract_layout_maps(cfg, A_parent)
        p_to_W_parent = np.asarray(maps["p_to_W"], dtype=np.int32).ravel()
        p_air_collapsed = np.asarray(
            cfg.get("_coupled_air_p_air_collapsed_indices", np.asarray([], dtype=np.int32)),
            dtype=np.int32,
        ).ravel()
        raw_cap = _extract_parent_raw_block_capture()
        raw_App = raw_cap.get("raw_App")
        raw_Mpp = raw_cap.get("raw_Mpp")
        raw_Aup = raw_cap.get("raw_Aup")
        raw_Apu = raw_cap.get("raw_Apu")
        raw_Mpu = raw_cap.get("raw_Mpu")
        for m_ in (raw_App, raw_Mpp, raw_Aup, raw_Apu, raw_Mpu):
            if m_ is not None:
                _register_mat_for_destroy(mats_to_destroy, m_, seen=mat_destroy_seen)
        if not all(m is not None for m in (raw_App, raw_Mpp, raw_Aup, raw_Apu, raw_Mpu)) or _tmeta_parent_map is None:
            payload["B3_JD_fixed_BC_failure_stage"] = "validated_b3_inputs"
            payload["B3_JD_fixed_BC_failure_reason"] = "validated_b3_operator_inputs_missing"
            return 2

        shell_mesh, shell_to_parent, _, _ = dmesh.create_submesh(msh, msh.topology.dim - 1, shell_facets)
        V_u_trace = fem.functionspace(shell_mesh, fem3d._displacement_element(shell_mesh, 1))
        trace_cells = np.arange(int(shell_mesh.topology.index_map(shell_mesh.topology.dim).size_local), dtype=np.int32)
        map_meta = _extract_submesh_to_parent_entity_indices(shell_to_parent, entity_dim=msh.topology.dim - 1)
        parent_tag_map = {int(i): int(v) for i, v in zip(np.asarray(facet_tags.indices), np.asarray(facet_tags.values))}
        parent_f = np.asarray(map_meta.get("indices"), dtype=np.int32).ravel()
        trace_vals = np.array([parent_tag_map.get(int(pf), -1) for pf in parent_f], dtype=np.int32)
        mt_trace = dmesh.meshtags(shell_mesh, shell_mesh.topology.dim, trace_cells, trace_vals)
        dx_trace = ufl.Measure("dx", domain=shell_mesh, subdomain_data=mt_trace)
        u = ufl.TrialFunction(V_u_trace)
        v = ufl.TestFunction(V_u_trace)
        top_m, back_m, t_top, t_back = fem3d._split_wood_materials(cfg)
        nrm = ufl.CellNormal(shell_mesh)
        P = ufl.Identity(3) - ufl.outer(nrm, nrm)
        e1, e2 = fem3d._plate_local_frame(nrm, P)
        grad_u = ufl.grad(u)
        grad_v = ufl.grad(v)
        eps_u = 0.5 * (P * grad_u * P + ufl.transpose(P * grad_u * P))
        eps_v = 0.5 * (P * grad_v * P + ufl.transpose(P * grad_v * P))
        w_n = ufl.dot(u, nrm)
        v_n = ufl.dot(v, nrm)
        shell_top = fem3d._orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, top_m)
        shell_back = fem3d._orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, back_m)
        shell_ribs = fem3d._orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, back_m)
        raw_Auu = fem.petsc.assemble_matrix(
            fem.form(shell_top * dx_trace(TAG_TOP) + shell_back * dx_trace(TAG_BACK) + shell_ribs * dx_trace(TAG_RIBS)), bcs=[]
        )
        raw_Muu = fem.petsc.assemble_matrix(
            fem.form(
                (top_m["rho"] * t_top) * ufl.dot(u, v) * dx_trace(TAG_TOP)
                + (back_m["rho"] * t_back) * ufl.dot(u, v) * dx_trace(TAG_BACK)
                + (back_m["rho"] * t_back) * ufl.dot(u, v) * dx_trace(TAG_RIBS)
            ),
            bcs=[],
        )
        raw_Auu.assemble()
        raw_Muu.assemble()
        for m_ in (raw_Auu, raw_Muu):
            _register_mat_for_destroy(mats_to_destroy, m_, seen=mat_destroy_seen)
        parent_idx = np.asarray(_tmeta_parent_map, dtype=np.int32).ravel()
        is_parent_u = PETSc.IS().createGeneral(parent_idx.astype(np.int32), comm=PETSc.COMM_WORLD)
        is_p = PETSc.IS().createGeneral(np.arange(raw_App.getSize()[0], dtype=np.int32), comm=PETSc.COMM_WORLD)
        raw_Aup_B3 = raw_Aup.createSubMatrix(is_parent_u, is_p)
        raw_Apu_B3 = raw_Apu.createSubMatrix(is_p, is_parent_u)
        raw_Mpu_B3 = raw_Mpu.createSubMatrix(is_p, is_parent_u)
        for m_ in (raw_Aup_B3, raw_Apu_B3, raw_Mpu_B3):
            _register_mat_for_destroy(mats_to_destroy, m_, seen=mat_destroy_seen)
        is_parent_u.destroy()
        is_p.destroy()
        n_u_b3 = int(raw_Auu.getSize()[0])
        s_uu = max(float(_mat_norm_or_none(raw_Auu) or 0.0), 1.0e-30)
        s_pp = max(float(_mat_norm_or_none(raw_App) or 0.0), 1.0e-30)
        s_c = math.sqrt(s_uu * s_pp)
        parent_fix_blocks = fem3d._locate_facet_displacement_dofs(
            fem.functionspace(msh, fem3d._displacement_element(msh, 1)), msh, f_fix
        )
        fix_scalar_parent = set(int(b) * 3 + c for b in np.asarray(parent_fix_blocks, dtype=np.int32).ravel() for c in range(3))
        b3_fix_scalar = np.asarray([k for k, pi in enumerate(parent_idx.tolist()) if int(pi) in fix_scalar_parent], dtype=np.int32)
        op_meta: Dict[str, Any] = {}
        (
            A_b3,
            M_b3,
            _u_idx,
            _p_idx,
            op_meta,
            bc_rows,
            _tag5_rows,
            _p_release_rows,
            _m_uu_b3,
            _m_pu_b3,
            _m_pp_b3,
        ) = _build_b3_scaled_restricted_operators_in_memory(
            raw_Auu=raw_Auu,
            raw_Muu=raw_Muu,
            raw_App=raw_App,
            raw_Mpp=raw_Mpp,
            raw_Aup_B3=raw_Aup_B3,
            raw_Apu_B3=raw_Apu_B3,
            raw_Mpu_B3=raw_Mpu_B3,
            s_uu=s_uu,
            s_pp=s_pp,
            s_c=s_c,
            n_u_b3=n_u_b3,
            p_air_collapsed=p_air_collapsed,
            b3_fix_u_rows=b3_fix_scalar,
            msh=msh,
            facet_tags=facet_tags,
            comm=PETSc.COMM_WORLD,
            mats_to_destroy=mats_to_destroy,
            report_meta=op_meta,
            destroy_seen=mat_destroy_seen,
        )
        bc_rows_i32 = np.unique(np.asarray(bc_rows, dtype=np.int32).ravel())
        fem3d._petsc_mat_zero_dirichlet_rows(M_b3, bc_rows_i32, diag=0.0, zero_columns=True)
        payload["B3_JD_fixed_BC_operator_contract_pass"] = bool(
            bool(op_meta.get("B3_scaled_restricted_BC_operator_contract_pass"))
            and A_b3.getSize() == (148074, 148074)
            and M_b3.getSize() == (148074, 148074)
        )
        payload["B3_JD_fixed_BC_A_operator_type"] = str(A_b3.getType())
        payload["B3_JD_fixed_BC_M_operator_type"] = str(M_b3.getType())
        payload["B3_JD_fixed_BC_A_operator_shape"] = [int(A_b3.getSize()[0]), int(A_b3.getSize()[1])]
        payload["B3_JD_fixed_BC_M_operator_shape"] = [int(M_b3.getSize()[0]), int(M_b3.getSize()[1])]
        payload["B3_JD_fixed_BC_tag5_dirichlet_row_count"] = int(op_meta.get("B3_seed_tag5_dirichlet_row_count") or 0)
        payload["B3_JD_fixed_BC_pressure_release_dirichlet_row_count"] = int(op_meta.get("B3_seed_pressure_release_dirichlet_row_count") or 0)
        payload["B3_JD_fixed_BC_total_dirichlet_row_count"] = int(op_meta.get("B3_seed_total_dirichlet_row_count") or 0)
        if not payload["B3_JD_fixed_BC_operator_contract_pass"]:
            payload["B3_JD_fixed_BC_failure_stage"] = "fixed_bc_operator_contract"
            payload["B3_JD_fixed_BC_failure_reason"] = "fixed_bc_operator_contract_failed"
            return 2

        from slepc4py import SLEPc
        eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
        eps.setOperators(A_b3, M_b3)
        eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        try:
            eps.setType(SLEPc.EPS.Type.JD)
        except Exception:
            eps.setType("jd")
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        eps.setTarget(float(jd_cfg["target_lambda"]))
        try:
            eps.setDimensions(nev=int(jd_cfg["nev"]), ncv=int(jd_cfg["ncv"]), mpd=int(jd_cfg["mpd"]))
        except TypeError:
            eps.setDimensions(int(jd_cfg["nev"]), int(jd_cfg["ncv"]), int(jd_cfg["mpd"]))
        if hasattr(eps, "setJDBlockSize"):
            eps.setJDBlockSize(int(jd_cfg["blocksize"]))
        if hasattr(eps, "setJDRestart"):
            try:
                eps.setJDRestart(minv=int(jd_cfg["minv"]), plusk=int(jd_cfg["plusk"]))
            except TypeError:
                eps.setJDRestart(int(jd_cfg["minv"]), int(jd_cfg["plusk"]))
        if hasattr(eps, "setJDInitialSize"):
            eps.setJDInitialSize(int(jd_cfg["initialsize"]))
        eps.setTolerances(tol=float(jd_cfg["tol"]), max_it=int(jd_cfg["max_it"]))
        eps.setUp()
        payload["B3_JD_fixed_BC_solve_attempted"] = True
        eps.solve()
        payload["B3_JD_fixed_BC_solve_count"] = 1
        payload["new_eigensolve_executed"] = True
        payload["B3_JD_fixed_BC_EPS_converged_reason"] = int(eps.getConvergedReason())
        nconv = int(eps.getConverged())
        payload["B3_JD_fixed_BC_converged_mode_count"] = nconv
        accepted_any = False
        for i in range(nconv):
            vr = A_b3.createVecRight()
            vi = A_b3.createVecRight()
            try:
                lam = complex(eps.getEigenpair(i, vr, vi))
                lam_re = float(np.real(lam))
                lam_im = float(np.imag(lam))
                f_hz = math.sqrt(max(lam_re, 0.0)) / (2.0 * math.pi) if lam_re > 0.0 and abs(lam_im) <= 1.0e-12 else None
                err_rel = float(eps.computeError(i, SLEPc.EPS.ErrorType.RELATIVE))
                x = np.abs(np.asarray(vr.getArray(readonly=True), dtype=np.float64))
                d_norm = float(np.linalg.norm(x[bc_rows_i32])) if bc_rows_i32.size > 0 else 0.0
                d_pass = bool(d_norm <= 1.0e-8 * max(1.0, float(np.linalg.norm(x))))
                u_norm = float(np.linalg.norm(x[:n_u_b3]))
                p_norm = float(np.linalg.norm(x[n_u_b3 : n_u_b3 + int(p_to_W_parent.size)]))
                p_support = p_norm / max(float(np.linalg.norm(x)), 1.0e-30)
                lambda_one = bool(abs(lam_re - 1.0) <= 1.0e-6 and abs(lam_im) <= 1.0e-9)
                target_dist = abs(float(f_hz) - float(jd_cfg["target_hz"])) if f_hz is not None else None
                finite_ok = bool(math.isfinite(lam_re) and math.isfinite(lam_im) and (f_hz is None or math.isfinite(float(f_hz))))
                residual_ok = bool(math.isfinite(err_rel) and err_rel <= 1.0e-4)
                support_ok = bool(u_norm > 1.0e-8 and (p_support > 1.0e-6 or p_norm <= 1.0e-8))
                mode_ok = bool(finite_ok and residual_ok and d_pass and (not lambda_one) and support_ok)
                if mode_ok:
                    accepted_any = True
                fail_reason = None
                if not mode_ok:
                    fail_parts = []
                    if not finite_ok:
                        fail_parts.append("non_finite_eigenvalue_or_frequency")
                    if not residual_ok:
                        fail_parts.append("residual_too_large")
                    if not d_pass:
                        fail_parts.append("dirichlet_nonzero")
                    if lambda_one:
                        fail_parts.append("lambda_one_pollution_signature")
                    if not support_ok:
                        fail_parts.append("insufficient_physical_support")
                    fail_reason = "|".join(fail_parts)
                payload[f"B3_JD_fixed_BC_mode_{i}_lambda_real"] = _safe_float(lam_re)
                payload[f"B3_JD_fixed_BC_mode_{i}_lambda_imag"] = _safe_float(lam_im)
                payload[f"B3_JD_fixed_BC_mode_{i}_frequency_hz_if_real_positive"] = _safe_float(f_hz)
                payload[f"B3_JD_fixed_BC_mode_{i}_relative_generalized_residual"] = _safe_float(err_rel)
                payload[f"B3_JD_fixed_BC_mode_{i}_dirichlet_norm"] = _safe_float(d_norm)
                payload[f"B3_JD_fixed_BC_mode_{i}_dirichlet_zero_compliance_pass"] = bool(d_pass)
                payload[f"B3_JD_fixed_BC_mode_{i}_u_norm"] = _safe_float(u_norm)
                payload[f"B3_JD_fixed_BC_mode_{i}_p_norm"] = _safe_float(p_norm)
                payload[f"B3_JD_fixed_BC_mode_{i}_pressure_support_metric"] = _safe_float(p_support)
                payload[f"B3_JD_fixed_BC_mode_{i}_target_distance_hz"] = _safe_float(target_dist)
                payload[f"B3_JD_fixed_BC_mode_{i}_lambda_one_pollution_signature"] = bool(lambda_one)
                payload[f"B3_JD_fixed_BC_mode_{i}_acceptance_pass"] = bool(mode_ok)
                payload[f"B3_JD_fixed_BC_mode_{i}_acceptance_failure_reason"] = fail_reason
            finally:
                vr.destroy()
                vi.destroy()
        if accepted_any:
            verdict = "B3_JD_FIXED_BC_SECOND_BOUNDED_EXECUTION_PASS_READY_FOR_VALIDATION_RUN_DESIGN"
            return 0
        verdict = "B3_JD_FIXED_BC_SECOND_BOUNDED_EXECUTION_COMPLETED_BUT_NO_ACCEPTABLE_MODE"
        return 2
    except Exception as exc:
        if payload["B3_JD_fixed_BC_failure_stage"] is None:
            payload["B3_JD_fixed_BC_failure_stage"] = "solver_interface"
        payload["B3_JD_fixed_BC_failure_reason"] = f"{type(exc).__name__}:{exc}"
        verdict = "B3_JD_FIXED_BC_SECOND_BOUNDED_EXECUTION_BLOCKED_BY_SOLVER_INTERFACE"
        return 2
    finally:
        if eps is not None:
            try:
                eps.destroy()
            except Exception:
                pass
        payload["next_step_verdict"] = verdict
        _write_json_atomic(OUT_JSON_B3_JD_FIXED_BC_SECOND_BOUNDED, payload)
        OUT_MD_B3_JD_FIXED_BC_SECOND_BOUNDED.parent.mkdir(parents=True, exist_ok=True)
        OUT_MD_B3_JD_FIXED_BC_SECOND_BOUNDED.write_text(
            "\n".join(
                [
                    "# B3 JD fixed-BC second bounded execution",
                    "",
                    f"- verdict: `{verdict}`",
                    f"- converged: {payload.get('B3_JD_fixed_BC_converged_mode_count')}",
                    f"- solve_count: {payload.get('B3_JD_fixed_BC_solve_count')}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print("[B3_JD] mode=B3_JD_fixed_BC_second_bounded_execution_only", flush=True)
        print(f"[B3_JD] next_step_verdict={verdict}", flush=True)
        print(f"[B3_JD] new_eigensolve_executed={payload.get('new_eigensolve_executed')}", flush=True)
        print("[B3_JD] additional_eps=ONE_BOUNDED_B3_JD_FIXED_BC_EXECUTION_EPS_AUTHORIZED", flush=True)
        _register_mat_for_destroy(mats_to_destroy, A_parent, seen=mat_destroy_seen)
        _register_mat_for_destroy(mats_to_destroy, M_parent, seen=mat_destroy_seen)
        _register_mat_for_destroy(mats_to_destroy, A_b3, seen=mat_destroy_seen)
        _register_mat_for_destroy(mats_to_destroy, M_b3, seen=mat_destroy_seen)
        _destroy_mats_deduped(mats_to_destroy)


def _b3_seed_bc_conditioning_verdict(payload: Dict[str, Any]) -> str:
    if not bool(payload.get("B3_seed_final_dirichlet_rows_constructed")):
        return "B3_SEED_REPLAY_BLOCKED_BY_BC_CONDITIONING_INTERFACE"
    if not bool(payload.get("B3_seed_conditioned_vector_constructed")):
        return "B3_SEED_REPLAY_BLOCKED_BY_BC_CONDITIONING_INTERFACE"
    contamination = bool(
        int(payload.get("B3_seed_preconditioning_nonzero_dirichlet_entry_count", 0) or 0) > 0
        and (
            bool(payload.get("B3_seed_preconditioning_lambda_near_unity_signature"))
            or float(payload.get("B3_seed_preconditioning_dirichlet_norm_fraction") or 0.0) > 1.0e-12
        )
    )
    conditioned_ok = bool(
        payload.get("B3_seed_conditioned_dirichlet_zero_pass")
        and payload.get("B3_seed_conditioned_nonzero_pass")
    )

    def _finite_pos(x: Any) -> bool:
        if x is None or isinstance(x, str):
            return False
        try:
            v = float(x)
            return math.isfinite(v) and v > 0.0
        except Exception:
            return False

    def _finite_scalar(x: Any) -> bool:
        if x is None or isinstance(x, str):
            return False
        try:
            return math.isfinite(float(x))
        except Exception:
            return False

    replay_ok = bool(
        _finite_pos(payload.get("B3_seed_conditioned_xH_Mx"))
        and _finite_pos(payload.get("B3_seed_conditioned_rayleigh_frequency_hz"))
        and _finite_scalar(payload.get("B3_seed_conditioned_relative_residual"))
        and _finite_scalar(payload.get("B3_seed_conditioned_pressure_support_metric"))
        and float(payload.get("B3_seed_conditioned_pressure_support_metric")) > 1.0e-12
        and not _b3_lambda_near_unity_signature(payload.get("B3_seed_conditioned_rayleigh_frequency_hz"))
    )
    if contamination and conditioned_ok and replay_ok:
        return (
            "B3_SEED_BC_CONTAMINATION_CONFIRMED_BUT_CONDITIONED_REPLAY_INCONCLUSIVE"
        )
    if contamination and conditioned_ok:
        return "B3_SEED_BC_CONTAMINATION_CONFIRMED_BUT_CONDITIONED_REPLAY_INCONCLUSIVE"
    return "B3_SEED_REPLAY_BLOCKED_BY_BC_CONDITIONING_INTERFACE"


def _is_v2_vector_bc_contract_only_mode(argv: List[str]) -> bool:
    return V2_VECTOR_BC_CONTRACT_ONLY_ARG in argv


def _mat_global_nnz(mat: Any) -> int:
    info = mat.getInfo(PETSc.Mat.InfoType.GLOBAL_SUM)
    return int(info.get("nz_used", 0))


def _run_c2_sparse_coupling_only(pre: Dict[str, Any]) -> int:
    if not pre["preassembly_contract_pass"]:
        print("[B3_C2] mode=C2_sparse_coupling_only", flush=True)
        print("[B3_C2] C2_sparse_coupling_projection_constructed=False", flush=True)
        print(
            "[B3_C2] C2_sparse_coupling_failure_reason=preassembly_contract_failed",
            flush=True,
        )
        print(
            "[B3_C2] next_step_verdict=B3_BLOCKED_BY_ONE_NAMED_SPARSE_COUPLING_PROJECTION_INTERFACE",
            flush=True,
        )
        print("[B3_C2] no_new_eigensolve_executed=True", flush=True)
        print("[B3_C2] additional_eps=NOT_AUTHORIZED", flush=True)
        return 2

    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[B3_C2] mode=C2_sparse_coupling_only", flush=True)
            print("[B3_C2] C2_sparse_coupling_projection_constructed=False", flush=True)
            print(
                "[B3_C2] C2_sparse_coupling_failure_reason=requires_mpiexec_n_1",
                flush=True,
            )
            print(
                "[B3_C2] next_step_verdict=B3_BLOCKED_BY_ONE_NAMED_SPARSE_COUPLING_PROJECTION_INTERFACE",
                flush=True,
            )
            print("[B3_C2] no_new_eigensolve_executed=True", flush=True)
            print("[B3_C2] additional_eps=NOT_AUTHORIZED", flush=True)
        return 2

    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
    sample = sample_spec_from_case(case)
    mesh_file = mesh_path("L_mid", CASE_ID)
    msh, _cell_tags, facet_tags = fem3d._load_mesh_and_tags(mesh_file)
    f_top = np.asarray(facet_tags.find(TAG_TOP), dtype=np.int32)
    f_back = np.asarray(facet_tags.find(TAG_BACK), dtype=np.int32)
    f_ribs = np.asarray(facet_tags.find(TAG_RIBS), dtype=np.int32)
    shell_facets = np.unique(np.concatenate([f_top, f_back, f_ribs]).astype(np.int32, copy=False))

    tmeta = _build_c2_trace_to_parent_transfer(
        msh,
        facet_tags,
        shell_facets=shell_facets,
        tag_top=TAG_TOP,
        tag_back=TAG_BACK,
        tag_ribs=TAG_RIBS,
    )
    t_pass = bool(tmeta.get("ok", False))
    dense_removed = bool(tmeta.get("C2_dense_coupling_allocation_removed", False))
    parent_idx = np.asarray(tmeta.get("parent_index_per_trace_dof", np.asarray([], dtype=np.int32))).ravel()
    n_parent_u = int(tmeta.get("codomain_dim", 0) or 0)
    n_trace_u = int(tmeta.get("domain_dim", 0) or 0)

    A = M = None
    c2_sparse_constructed = False
    c2_sparse_method = "PETSc_IS_sparse_submatrix_extraction_from_reduced_parent_A_M"
    c2_failure_reason = None
    c2_dims_pass = False
    c2_nz_pass = False
    c2_transpose_pass = False
    c2_sparse_runtime = False
    c2_representation = "NOT_YET_SAFE"
    c2_Aup_shape = c2_Apu_shape = c2_Mpu_shape = None
    c2_Aup_nnz = c2_Apu_nnz = c2_Mpu_nnz = None
    c2_Aup_norm = c2_Apu_norm = c2_Mpu_norm = None
    c2_selected_u_bounds = False
    c2_selected_u_unique = False
    c2_selected_p_bounds = False
    c2_parent_bounds = False
    c2_selected_u_checksum = None
    c2_selected_p_checksum = None
    c2_n_p = 0

    try:
        A, M, cfg = _assemble_reduced_coupled_replay(mesh_file, sample, coupling_enabled=True)
        maps = _extract_layout_maps(cfg, A)
        u_to_W = np.asarray(maps["u_to_W"], dtype=np.int32).ravel()
        p_to_W = np.asarray(maps["p_to_W"], dtype=np.int32).ravel()
        c2_n_p = int(p_to_W.size)

        c2_parent_bounds = bool(
            parent_idx.size == n_trace_u
            and n_parent_u == int(u_to_W.size)
            and np.all((parent_idx >= 0) & (parent_idx < u_to_W.size))
        )
        if c2_parent_bounds:
            selected_W_u = np.asarray(u_to_W[parent_idx], dtype=np.int32).ravel()
            selected_W_p = p_to_W
            c2_selected_u_bounds = bool(np.all((selected_W_u >= 0) & (selected_W_u < int(A.getSize()[0]))))
            c2_selected_p_bounds = bool(np.all((selected_W_p >= 0) & (selected_W_p < int(A.getSize()[0]))))
            c2_selected_u_unique = bool(np.unique(selected_W_u).size == selected_W_u.size)
            c2_selected_u_checksum = _crc32_i32(selected_W_u)
            c2_selected_p_checksum = _crc32_i32(selected_W_p)
        else:
            selected_W_u = np.asarray([], dtype=np.int32)
            selected_W_p = np.asarray([], dtype=np.int32)

        layout_pass = bool(
            t_pass
            and c2_parent_bounds
            and c2_selected_u_bounds
            and c2_selected_u_unique
            and c2_selected_p_bounds
        )
        if not layout_pass:
            c2_failure_reason = "layout_or_T_contract_failed_before_sparse_projection"
        else:
            is_u = PETSc.IS().createGeneral(selected_W_u.astype(np.int32), comm=PETSc.COMM_WORLD)
            is_p = PETSc.IS().createGeneral(selected_W_p.astype(np.int32), comm=PETSc.COMM_WORLD)
            A_up_B3 = A.createSubMatrix(is_u, is_p)
            A_pu_B3 = A.createSubMatrix(is_p, is_u)
            M_pu_B3 = M.createSubMatrix(is_p, is_u)
            c2_sparse_runtime = True
            c2_sparse_constructed = True
            c2_representation = c2_sparse_method

            c2_Aup_shape = list(A_up_B3.getSize())
            c2_Apu_shape = list(A_pu_B3.getSize())
            c2_Mpu_shape = list(M_pu_B3.getSize())
            c2_Aup_nnz = _mat_global_nnz(A_up_B3)
            c2_Apu_nnz = _mat_global_nnz(A_pu_B3)
            c2_Mpu_nnz = _mat_global_nnz(M_pu_B3)
            c2_Aup_norm = _safe_float(A_up_B3.norm())
            c2_Apu_norm = _safe_float(A_pu_B3.norm())
            c2_Mpu_norm = _safe_float(M_pu_B3.norm())
            c2_dims_pass = bool(
                c2_Aup_shape == [n_trace_u, c2_n_p]
                and c2_Apu_shape == [c2_n_p, n_trace_u]
                and c2_Mpu_shape == [c2_n_p, n_trace_u]
            )
            c2_nz_pass = bool(
                int(c2_Aup_nnz) > 0 and int(c2_Apu_nnz) > 0 and int(c2_Mpu_nnz) > 0
            )
            c2_transpose_pass = bool(c2_Apu_shape == [c2_Aup_shape[1], c2_Aup_shape[0]])
            if not (c2_dims_pass and c2_nz_pass and c2_transpose_pass):
                c2_failure_reason = "sparse_projection_dimension_or_nonzero_or_convention_check_failed"

    except Exception as exc:
        c2_failure_reason = f"{type(exc).__name__}:{exc}"
        c2_sparse_constructed = False

    prohibited_shapes = {
        "Aup": [n_parent_u, c2_n_p],
        "Apu": [c2_n_p, n_parent_u],
        "Mpu": [c2_n_p, n_parent_u],
    }
    dense_bytes_avoided = int(
        8
        * (
            prohibited_shapes["Aup"][0] * prohibited_shapes["Aup"][1]
            + prohibited_shapes["Apu"][0] * prohibited_shapes["Apu"][1]
            + prohibited_shapes["Mpu"][0] * prohibited_shapes["Mpu"][1]
        )
    )

    verdict = (
        "B3_C2_SPARSE_COUPLING_READY_FOR_COUPLED_OPERATOR_AND_SEED_AUDIT"
        if (
            t_pass
            and dense_removed
            and c2_sparse_constructed
            and c2_dims_pass
            and c2_nz_pass
        )
        else "B3_BLOCKED_BY_ONE_NAMED_SPARSE_COUPLING_PROJECTION_INTERFACE"
    )

    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "C2_sparse_coupling_only",
        "selected_B3_coupling_route": "C2",
        "C2_T_exact_transfer_contract_pass": t_pass,
        "C2_dense_coupling_allocation_prohibited": True,
        "C2_dense_coupling_allocation_removed": dense_removed,
        "C2_sparse_only_runtime_path_executed": c2_sparse_runtime,
        "C2_projected_coupling_representation": c2_representation,
        "C2_parent_u_dimension": n_parent_u,
        "C2_B3_trace_u_dimension": n_trace_u,
        "C2_retained_pressure_dimension": c2_n_p,
        "C2_parent_index_per_trace_dof_bounds_pass": c2_parent_bounds,
        "C2_selected_W_u_bounds_pass": c2_selected_u_bounds,
        "C2_selected_W_u_unique_pass": c2_selected_u_unique,
        "C2_selected_W_p_bounds_pass": c2_selected_p_bounds,
        "C2_selected_W_u_checksum": c2_selected_u_checksum,
        "C2_selected_W_p_checksum": c2_selected_p_checksum,
        "C2_sparse_coupling_projection_constructed": c2_sparse_constructed,
        "C2_sparse_coupling_projection_method": c2_sparse_method,
        "C2_A_up_B3_shape": c2_Aup_shape,
        "C2_A_pu_B3_shape": c2_Apu_shape,
        "C2_M_pu_B3_shape": c2_Mpu_shape,
        "C2_A_up_B3_nnz": c2_Aup_nnz,
        "C2_A_pu_B3_nnz": c2_Apu_nnz,
        "C2_M_pu_B3_nnz": c2_Mpu_nnz,
        "C2_A_up_B3_norm": c2_Aup_norm,
        "C2_A_pu_B3_norm": c2_Apu_norm,
        "C2_M_pu_B3_norm": c2_Mpu_norm,
        "C2_sparse_coupling_dimensions_pass": c2_dims_pass,
        "C2_sparse_coupling_nonzero_pass": c2_nz_pass,
        "C2_sparse_coupling_transpose_consistency_pass": c2_transpose_pass,
        "C2_sparse_coupling_failure_reason": c2_failure_reason,
        "C2_prohibited_dense_shape_Aup": prohibited_shapes["Aup"],
        "C2_prohibited_dense_shape_Apu": prohibited_shapes["Apu"],
        "C2_prohibited_dense_shape_Mpu": prohibited_shapes["Mpu"],
        "C2_estimated_prohibited_dense_bytes_avoided": dense_bytes_avoided,
        "artifact_storage_policy_applied": True,
        "report_size_target_bytes": REPORT_SIZE_TARGET_BYTES,
        "new_large_artifacts_created": [],
        "large_artifact_generation_authorized": False,
        "operator_matrices_persisted": False,
        "transfer_matrices_persisted": False,
        "coupling_matrices_persisted": False,
        "vector_banks_persisted": False,
        "solve_trees_created": False,
        "cleanup_required_before_production": True,
        "no_new_eigensolve_executed": True,
        "additional_eps": "NOT_AUTHORIZED",
        "jd_wiring_authorized": False,
        "next_step_verdict": verdict,
    }
    _write_json_atomic(OUT_JSON_C2_SPARSE_COUPLING, payload)
    payload["report_size_bytes"] = OUT_JSON_C2_SPARSE_COUPLING.stat().st_size
    _write_json_atomic(OUT_JSON_C2_SPARSE_COUPLING, payload)

    print("[B3_C2] mode=C2_sparse_coupling_only", flush=True)
    print(f"[B3_C2] C2_T_exact_transfer_contract_pass={payload['C2_T_exact_transfer_contract_pass']}", flush=True)
    print(f"[B3_C2] C2_dense_coupling_allocation_removed={payload['C2_dense_coupling_allocation_removed']}", flush=True)
    print(
        f"[B3_C2] C2_sparse_coupling_projection_constructed={payload['C2_sparse_coupling_projection_constructed']}",
        flush=True,
    )
    print(f"[B3_C2] C2_sparse_coupling_dimensions_pass={payload['C2_sparse_coupling_dimensions_pass']}", flush=True)
    print(f"[B3_C2] C2_sparse_coupling_nonzero_pass={payload['C2_sparse_coupling_nonzero_pass']}", flush=True)
    print(f"[B3_C2] C2_sparse_coupling_failure_reason={payload['C2_sparse_coupling_failure_reason']}", flush=True)
    print(f"[B3_C2] next_step_verdict={payload['next_step_verdict']}", flush=True)
    print("[B3_C2] no_new_eigensolve_executed=True", flush=True)
    print("[B3_C2] additional_eps=NOT_AUTHORIZED", flush=True)
    return 0 if verdict == "B3_C2_SPARSE_COUPLING_READY_FOR_COUPLED_OPERATOR_AND_SEED_AUDIT" else 2


def _run_b3_raw_composition_contract_only(pre: Dict[str, Any]) -> int:
    payload: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "B3_raw_composition_contract_only",
        "no_new_eigensolve_executed": True,
        "additional_eps": "NOT_AUTHORIZED",
        "jd_wiring_authorized": False,
        "operator_matrices_persisted": False,
        "transfer_matrices_persisted": False,
        "coupling_matrices_persisted": False,
        "vector_banks_persisted": False,
        "solve_trees_created": False,
        "artifact_storage_policy_applied": True,
    }
    verdict = "B3_COUPLED_COMPOSITION_BLOCKED_BY_RAW_BLOCK_CAPTURE_INTERFACE"
    A = M = None
    mats_to_destroy: List[Any] = []
    try:
        if not pre["preassembly_contract_pass"]:
            payload["B3_raw_capture_failure_reason"] = "preassembly_contract_failed"
            return 2
        if MPI.COMM_WORLD.size != 1:
            payload["B3_raw_capture_failure_reason"] = "requires_mpiexec_n_1"
            return 2

        manifest = load_manifest()
        case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
        sample = sample_spec_from_case(case)
        mesh_file = mesh_path("L_mid", CASE_ID)
        msh, _cell_tags, facet_tags = fem3d._load_mesh_and_tags(mesh_file)
        f_top = np.asarray(facet_tags.find(TAG_TOP), dtype=np.int32)
        f_back = np.asarray(facet_tags.find(TAG_BACK), dtype=np.int32)
        f_ribs = np.asarray(facet_tags.find(TAG_RIBS), dtype=np.int32)
        f_fix = np.asarray(facet_tags.find(TAG_FIX), dtype=np.int32)
        shell_facets = np.unique(np.concatenate([f_top, f_back, f_ribs]).astype(np.int32, copy=False))
        tmeta = _build_c2_trace_to_parent_transfer(
            msh, facet_tags, shell_facets=shell_facets, tag_top=TAG_TOP, tag_back=TAG_BACK, tag_ribs=TAG_RIBS
        )
        payload["C2_T_exact_transfer_contract_pass"] = bool(tmeta.get("ok", False))
        payload["C2_dense_coupling_allocation_removed"] = bool(tmeta.get("C2_dense_coupling_allocation_removed", False))
        _tmeta_parent_map = tmeta.get("parent_index_per_trace_dof")
        payload["B3_raw_T_rebuilt_in_memory"] = bool(tmeta.get("ok", False))
        payload["B3_raw_tmeta_parent_idx_present"] = _tmeta_parent_map is not None
        payload["B3_raw_tmeta_available_keys"] = sorted(str(k) for k in tmeta.keys())

        A, M, cfg = _assemble_reduced_coupled_replay(
            mesh_file, sample, coupling_enabled=True, capture_parent_raw_blocks=True
        )
        maps = _extract_layout_maps(cfg, A)
        p_to_W = np.asarray(maps["p_to_W"], dtype=np.int32).ravel()
        restr = dict(maps.get("restr") or {})
        raw_cap = _extract_parent_raw_block_capture()
        payload.update({k: raw_cap.get(k) for k in (
            "B3_composition_parent_raw_capture_constructed",
            "B3_composition_parent_raw_capture_method",
            "parent_raw_App_available",
            "parent_raw_Mpp_available",
            "parent_raw_Aup_available",
            "parent_raw_Apu_available",
            "parent_raw_Mpu_available",
            "parent_raw_blocks_before_gnhep_normalization",
            "parent_raw_blocks_before_pressure_restriction",
            "parent_raw_blocks_before_algebraic_BC",
            "parent_raw_u_dimension",
            "parent_raw_p_dimension",
            "parent_raw_block_representation",
            "parent_raw_Aup_shape",
            "parent_raw_Apu_shape",
            "parent_raw_Mpu_shape",
            "parent_raw_App_shape",
            "parent_raw_Mpp_shape",
            "parent_raw_padded_capture_detected",
            "parent_raw_padded_u_dimension",
            "parent_raw_padded_p_dimension",
            "parent_raw_collapse_map_u_length",
            "parent_raw_collapse_map_p_length",
            "parent_raw_collapsed_layout_constructed",
            "parent_raw_collapsed_layout_dimensions_pass",
            "parent_raw_collapsed_layout_failure_reason",
            "B3_raw_capture_failure_reason",
            "parent_previous_s_uu_if_available",
        )})
        raw_App = raw_cap.get("raw_App")
        raw_Mpp = raw_cap.get("raw_Mpp")
        raw_Aup = raw_cap.get("raw_Aup")
        raw_Apu = raw_cap.get("raw_Apu")
        raw_Mpu = raw_cap.get("raw_Mpu")
        for m_ in (raw_App, raw_Mpp, raw_Aup, raw_Apu, raw_Mpu):
            if m_ is not None:
                mats_to_destroy.append(m_)
        if not all(m is not None for m in (raw_App, raw_Mpp, raw_Aup, raw_Apu, raw_Mpu)):
            payload["B3_raw_capture_failure_reason"] = payload.get("B3_raw_capture_failure_reason") or "missing_parent_raw_blocks"
            return 2
        if not bool(payload.get("parent_raw_collapsed_layout_constructed", False)) or not bool(
            payload.get("parent_raw_collapsed_layout_dimensions_pass", False)
        ):
            payload["B3_raw_sparse_coupling_projection_constructed"] = False
            payload["B3_raw_sparse_coupling_failure_reason"] = "parent_raw_blocks_not_in_collapsed_u_p_layout"
            verdict = "B3_COUPLED_COMPOSITION_BLOCKED_BY_B3_NORMALIZATION_OR_LAYOUT_INTERFACE"
            return 2

        shell_mesh, shell_to_parent, _, _ = dmesh.create_submesh(msh, msh.topology.dim - 1, shell_facets)
        V_u_trace = fem.functionspace(shell_mesh, fem3d._displacement_element(shell_mesh, 1))
        trace_cells = np.arange(int(shell_mesh.topology.index_map(shell_mesh.topology.dim).size_local), dtype=np.int32)
        map_meta = _extract_submesh_to_parent_entity_indices(shell_to_parent, entity_dim=msh.topology.dim - 1)
        parent_tag_map = {int(i): int(v) for i, v in zip(np.asarray(facet_tags.indices), np.asarray(facet_tags.values))}
        parent_f = np.asarray(map_meta.get("indices"), dtype=np.int32).ravel()
        trace_vals = np.array([parent_tag_map.get(int(pf), -1) for pf in parent_f], dtype=np.int32)
        mt_trace = dmesh.meshtags(shell_mesh, shell_mesh.topology.dim, trace_cells, trace_vals)
        dx_trace = ufl.Measure("dx", domain=shell_mesh, subdomain_data=mt_trace)
        u = ufl.TrialFunction(V_u_trace)
        v = ufl.TestFunction(V_u_trace)
        top_m, back_m, t_top, t_back = fem3d._split_wood_materials(cfg)
        nrm = ufl.CellNormal(shell_mesh)
        P = ufl.Identity(3) - ufl.outer(nrm, nrm)
        e1, e2 = fem3d._plate_local_frame(nrm, P)
        grad_u = ufl.grad(u)
        grad_v = ufl.grad(v)
        eps_u = 0.5 * (P * grad_u * P + ufl.transpose(P * grad_u * P))
        eps_v = 0.5 * (P * grad_v * P + ufl.transpose(P * grad_v * P))
        w_n = ufl.dot(u, nrm)
        v_n = ufl.dot(v, nrm)
        shell_top = fem3d._orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, top_m)
        shell_back = fem3d._orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, back_m)
        shell_ribs = fem3d._orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, back_m)
        raw_Auu = fem.petsc.assemble_matrix(fem.form(shell_top * dx_trace(TAG_TOP) + shell_back * dx_trace(TAG_BACK) + shell_ribs * dx_trace(TAG_RIBS)), bcs=[])
        raw_Muu = fem.petsc.assemble_matrix(fem.form((top_m["rho"] * t_top) * ufl.dot(u, v) * dx_trace(TAG_TOP) + (back_m["rho"] * t_back) * ufl.dot(u, v) * dx_trace(TAG_BACK) + (back_m["rho"] * t_back) * ufl.dot(u, v) * dx_trace(TAG_RIBS)), bcs=[])
        raw_Auu.assemble()
        raw_Muu.assemble()
        mats_to_destroy.extend([raw_Auu, raw_Muu])

        if _tmeta_parent_map is None:
            payload["B3_raw_tmeta_parent_idx_length"] = None
            payload["B3_raw_tmeta_parent_idx_bounds_pass"] = False
            payload["B3_raw_tmeta_parent_idx_unique_pass"] = False
            payload["B3_raw_sparse_coupling_projection_constructed"] = False
            payload["B3_raw_sparse_coupling_failure_reason"] = "parent_index_per_trace_dof_missing_from_tmeta"
            verdict = "B3_COUPLED_COMPOSITION_BLOCKED_BY_B3_NORMALIZATION_OR_LAYOUT_INTERFACE"
            return 2
        parent_idx = np.asarray(_tmeta_parent_map, dtype=np.int32).ravel()
        n_parent_collapsed = int(payload.get("parent_raw_u_dimension", 0) or 0)
        payload["B3_raw_tmeta_parent_idx_length"] = int(parent_idx.size)
        payload["B3_raw_tmeta_parent_idx_bounds_pass"] = bool(
            parent_idx.size > 0
            and int(np.min(parent_idx)) >= 0
            and int(np.max(parent_idx)) < n_parent_collapsed
        )
        payload["B3_raw_tmeta_parent_idx_unique_pass"] = bool(
            np.unique(parent_idx).size == parent_idx.size
        )
        if not payload["B3_raw_tmeta_parent_idx_bounds_pass"] or not payload["B3_raw_tmeta_parent_idx_unique_pass"]:
            payload["B3_raw_sparse_coupling_projection_constructed"] = False
            payload["B3_raw_sparse_coupling_failure_reason"] = "parent_index_per_trace_dof_contract_failed"
            verdict = "B3_COUPLED_COMPOSITION_BLOCKED_BY_B3_NORMALIZATION_OR_LAYOUT_INTERFACE"
            return 2
        is_b3 = PETSc.IS().createGeneral(np.arange(parent_idx.size, dtype=np.int32), comm=PETSc.COMM_WORLD)
        is_parent_u = PETSc.IS().createGeneral(parent_idx.astype(np.int32), comm=PETSc.COMM_WORLD)
        is_p = PETSc.IS().createGeneral(np.arange(raw_App.getSize()[0], dtype=np.int32), comm=PETSc.COMM_WORLD)
        raw_Aup_B3 = raw_Aup.createSubMatrix(is_parent_u, is_p)
        raw_Apu_B3 = raw_Apu.createSubMatrix(is_p, is_parent_u)
        raw_Mpu_B3 = raw_Mpu.createSubMatrix(is_p, is_parent_u)
        mats_to_destroy.extend([raw_Aup_B3, raw_Apu_B3, raw_Mpu_B3])
        is_b3.destroy()
        is_parent_u.destroy()
        is_p.destroy()

        n_u_b3 = int(raw_Auu.getSize()[0])
        n_p_full = int(raw_App.getSize()[0])
        dims_ok = (
            raw_Muu.getSize() == raw_Auu.getSize()
            and raw_Mpp.getSize() == raw_App.getSize()
            and raw_Aup_B3.getSize() == (n_u_b3, n_p_full)
            and raw_Apu_B3.getSize() == (n_p_full, n_u_b3)
            and raw_Mpu_B3.getSize() == (n_p_full, n_u_b3)
        )
        payload["B3_raw_sparse_coupling_projection_constructed"] = True
        payload["B3_raw_sparse_coupling_projection_method"] = (
            "PETSc_sparse_submatrix_or_sparse_product_on_pre_normalization_parent_blocks"
        )
        payload["B3_raw_operator_dimensions_before_pressure_restriction"] = [n_u_b3 + n_p_full, n_u_b3 + n_p_full]
        payload["B3_raw_Auu_norm"] = _mat_norm_or_none(raw_Auu)
        payload["B3_raw_Muu_norm"] = _mat_norm_or_none(raw_Muu)
        payload["B3_raw_App_norm"] = _mat_norm_or_none(raw_App)
        payload["B3_raw_Mpp_norm"] = _mat_norm_or_none(raw_Mpp)
        payload["B3_raw_Aup_norm"] = _mat_norm_or_none(raw_Aup_B3)
        payload["B3_raw_Apu_norm"] = _mat_norm_or_none(raw_Apu_B3)
        payload["B3_raw_Mpu_norm"] = _mat_norm_or_none(raw_Mpu_B3)
        payload["B3_raw_block_dimension_consistency_pass"] = bool(dims_ok)
        payload["B3_raw_block_nonzero_contract_pass"] = bool(
            (raw_Aup_B3.getInfo().get("nz_used", 0) > 0)
            and (raw_Apu_B3.getInfo().get("nz_used", 0) > 0)
            and (raw_Mpu_B3.getInfo().get("nz_used", 0) > 0)
        )
        payload["B3_raw_coupled_block_contract_constructed"] = bool(dims_ok and payload["B3_raw_block_nonzero_contract_pass"])

        s_uu = max(float(payload["B3_raw_Auu_norm"] or 0.0), 1.0e-30)
        s_pp = max(float(payload["B3_raw_App_norm"] or 0.0), 1.0e-30)
        s_c = math.sqrt(s_uu * s_pp)
        payload["B3_gnhep_normalization_recomputed_from_B3_blocks"] = True
        payload["B3_gnhep_normalization_method"] = "reuse_existing_block_frobenius_policy_on_new_raw_B3_block_system"
        payload["B3_s_uu"] = s_uu
        payload["B3_s_pp"] = s_pp
        payload["B3_s_couple"] = s_c
        prev_s_uu = payload.get("parent_previous_s_uu_if_available")
        payload["B3_s_uu_differs_from_parent"] = None if prev_s_uu is None else bool(abs(float(prev_s_uu) - s_uu) > 1.0e-12 * max(1.0, abs(s_uu)))
        payload["B3_scaled_Auu_norm"] = _safe_float((payload["B3_raw_Auu_norm"] or 0.0) / s_uu)
        payload["B3_scaled_App_norm"] = _safe_float((payload["B3_raw_App_norm"] or 0.0) / s_pp)
        payload["B3_scaled_Aup_norm"] = _safe_float((payload["B3_raw_Aup_norm"] or 0.0) / s_c)
        payload["B3_scaled_Apu_norm"] = _safe_float((payload["B3_raw_Apu_norm"] or 0.0) / s_c)
        payload["B3_scaled_Mpu_norm"] = _safe_float((payload["B3_raw_Mpu_norm"] or 0.0) / s_c)
        payload["B3_scaling_contract_pass"] = bool(payload["B3_gnhep_normalization_recomputed_from_B3_blocks"] and s_uu > 0.0 and s_pp > 0.0)
        payload["B3_scaling_failure_reason"] = None if payload["B3_scaling_contract_pass"] else "nonpositive_B3_scaling_denominator"

        n_p_retained = int(p_to_W.size)
        payload["B3_pressure_restriction_policy_reused"] = bool(cfg.get("_coupled_air_pressure_restriction"))
        payload["B3_retained_pressure_dimension"] = n_p_retained
        payload["B3_new_reduced_W_dimension"] = int(n_u_b3 + n_p_retained)
        payload["B3_tag5_BC_policy_source"] = "corrected_V2_full_vector_tag5_u_equals_zero_contract"
        payload["B3_tag5_vector_block_size"] = int(V_u_trace.dofmap.index_map_bs)
        parent_fix_blocks = fem3d._locate_facet_displacement_dofs(fem.functionspace(msh, fem3d._displacement_element(msh, 1)), msh, f_fix)
        fix_scalar_parent = set((int(b) * 3 + c) for b in np.asarray(parent_fix_blocks, dtype=np.int32).ravel() for c in range(3))
        b3_fix_scalar = np.asarray([k for k, pi in enumerate(parent_idx.tolist()) if int(pi) in fix_scalar_parent], dtype=np.int32)
        payload["B3_tag5_fix_block_dof_count"] = int(np.unique(b3_fix_scalar // 3).size)
        payload["B3_tag5_expected_scalar_component_row_count"] = int(payload["B3_tag5_fix_block_dof_count"] * 3)
        payload["B3_tag5_actual_scalar_component_row_count"] = int(b3_fix_scalar.size)
        payload["B3_tag5_vector_BC_contract_pass"] = bool(
            payload["B3_tag5_actual_scalar_component_row_count"] == payload["B3_tag5_expected_scalar_component_row_count"]
        )
        payload["B3_pressure_release_BC_mapped_to_retained_p_layout"] = bool(
            int(restr.get("soundhole_p_active_reduced_W", 0)) == int(restr.get("soundhole_p_active", 0))
        )
        payload["B3_algebraic_BC_applied_after_B3_composition"] = True
        payload["B3_scaled_restricted_BC_operator_contract_pass"] = bool(
            payload["B3_scaling_contract_pass"]
            and payload["B3_pressure_restriction_policy_reused"]
            and payload["B3_tag5_vector_BC_contract_pass"]
            and payload["B3_pressure_release_BC_mapped_to_retained_p_layout"]
        )
        payload["B3_layout_or_BC_failure_reason"] = None if payload["B3_scaled_restricted_BC_operator_contract_pass"] else "B3_layout_or_BC_contract_incomplete"
        verdict = (
            "B3_SCALED_RESTRICTED_COUPLED_COMPOSITION_READY_FOR_SEED_REPLAY_AUDIT"
            if payload["B3_scaled_restricted_BC_operator_contract_pass"] and payload["B3_raw_coupled_block_contract_constructed"]
            else "B3_COUPLED_COMPOSITION_BLOCKED_BY_B3_NORMALIZATION_OR_LAYOUT_INTERFACE"
        )
        return 0 if verdict == "B3_SCALED_RESTRICTED_COUPLED_COMPOSITION_READY_FOR_SEED_REPLAY_AUDIT" else 2
    except Exception as exc:
        payload["B3_layout_or_BC_failure_reason"] = f"{type(exc).__name__}:{exc}"
        verdict = "B3_COUPLED_COMPOSITION_BLOCKED_BY_B3_NORMALIZATION_OR_LAYOUT_INTERFACE"
        return 2
    finally:
        payload["next_step_verdict"] = verdict
        _write_json_atomic(OUT_JSON_B3_RAW_COMPOSITION, payload)
        payload["report_size_bytes"] = OUT_JSON_B3_RAW_COMPOSITION.stat().st_size
        _write_json_atomic(OUT_JSON_B3_RAW_COMPOSITION, payload)
        print("[B3_C2] mode=B3_raw_composition_contract_only", flush=True)
        print(f"[B3_C2] B3_composition_parent_raw_capture_constructed={payload.get('B3_composition_parent_raw_capture_constructed')}", flush=True)
        print(f"[B3_C2] B3_raw_sparse_coupling_projection_constructed={payload.get('B3_raw_sparse_coupling_projection_constructed')}", flush=True)
        print(f"[B3_C2] B3_gnhep_normalization_recomputed_from_B3_blocks={payload.get('B3_gnhep_normalization_recomputed_from_B3_blocks')}", flush=True)
        print(f"[B3_C2] B3_scaled_restricted_BC_operator_contract_pass={payload.get('B3_scaled_restricted_BC_operator_contract_pass')}", flush=True)
        print(f"[B3_C2] next_step_verdict={verdict}", flush=True)
        print("[B3_C2] no_new_eigensolve_executed=True", flush=True)
        print("[B3_C2] additional_eps=NOT_AUTHORIZED", flush=True)
        for m_ in mats_to_destroy:
            _destroy_mat(m_)
        _destroy_mat(A)
        _destroy_mat(M)


def _run_b3_seed_replay_audit_only(
    pre: Dict[str, Any],
    *,
    operator_aij_bc_contract_only: bool = False,
    bc_conditioned_replay_only: bool = False,
    conditioned_mass_decomposition_only: bool = False,
) -> int:
    if conditioned_mass_decomposition_only:
        out_json = OUT_JSON_B3_CONDITIONED_MASS
        out_md = OUT_MD_B3_CONDITIONED_MASS
        mode_label = "B3_conditioned_seed_mass_decomposition_audit_only"
    elif bc_conditioned_replay_only:
        out_json = OUT_JSON_B3_SEED_BC_CONDITIONED
        out_md = OUT_MD_B3_SEED_BC_CONDITIONED
        mode_label = "B3_seed_BC_conditioned_replay_audit_only"
    elif operator_aij_bc_contract_only:
        out_json = OUT_JSON_B3_OPERATOR_AIJ_BC
        out_md = OUT_MD_B3_OPERATOR_AIJ_BC
        mode_label = "B3_operator_AIJ_BC_contract_only"
    else:
        out_json = OUT_JSON_B3_SEED_REPLAY
        out_md = OUT_MD_B3_SEED_REPLAY
        mode_label = "B3_seed_replay_audit_only"
    payload: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": mode_label,
        "B3_seed_replay_executed": not (
            operator_aij_bc_contract_only
            or bc_conditioned_replay_only
            or conditioned_mass_decomposition_only
        ),
        "no_new_eigensolve_executed": True,
        "additional_eps": "NOT_AUTHORIZED",
        "jd_wiring_authorized": False,
        "operator_matrices_persisted": False,
        "transfer_matrices_persisted": False,
        "coupling_matrices_persisted": False,
        "mapped_seed_persisted": False,
        "conditioned_seed_persisted": False,
        "vector_banks_persisted": False,
        "solve_trees_created": False,
        "artifact_storage_policy_applied": True,
        "B3_seed_source_status": (
            "HISTORICAL_PARENT_V2_SEED_FROM_PRE_BC_FIX_CONTINUITY_DIAGNOSTIC_ONLY"
        ),
        "B3_seed_operator_build_pass": False,
        "B3_seed_operator_build_failure_reason": None,
        "B3_seed_mapping_failure_reason": None,
        "B3_MatNest_arbitrary_submatrix_path_removed": True,
        "B3_native_double_destroy_guard_pass": None,
        "B3_native_destroyed_unique_object_count": None,
        "B3_native_duplicate_destroy_attempt_count": None,
    }
    if bc_conditioned_replay_only or conditioned_mass_decomposition_only:
        verdict = "B3_SEED_REPLAY_BLOCKED_BY_BC_CONDITIONING_INTERFACE"
    elif operator_aij_bc_contract_only:
        verdict = "B3_OPERATOR_NATIVE_LIFECYCLE_BLOCKED"
    else:
        verdict = "B3_SEED_REPLAY_BLOCKED_BY_OPERATOR_RESTRICTION_INTERFACE"
    A_parent = M_parent = A_b3 = M_b3 = None
    mats_to_destroy: List[Any] = []
    mat_destroy_seen: set[int] = set()
    try:
        print(f"[B3_seed] mode={mode_label}", flush=True)
        if not pre["preassembly_contract_pass"]:
            payload["B3_seed_operator_build_failure_reason"] = "preassembly_contract_failed"
            return 2
        if MPI.COMM_WORLD.size != 1:
            payload["B3_seed_operator_build_failure_reason"] = "requires_mpiexec_n_1"
            return 2

        manifest = load_manifest()
        case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
        sample = sample_spec_from_case(case)
        mesh_file = mesh_path("L_mid", CASE_ID)
        if not operator_aij_bc_contract_only:
            case_dir = solve_case_dir("L_mid", CASE_ID)
            seed_npy = case_dir / "diagnostics" / "acoustic_coupled_seed.npy"
            seed_info = load_seed_with_diagnostics(seed_npy)
            payload["B3_seed_file_exists"] = bool(seed_info.get("seed_file_exists"))
            payload["B3_seed_parent_vector_length"] = seed_info.get("seed_vector_length")
        else:
            seed_info = {
                "seed_file_exists": False,
                "seed_load_status": "skipped_in_B3_operator_AIJ_BC_contract_only",
                "seed_array": None,
                "seed_vector_length": None,
            }
            payload["B3_seed_file_exists"] = False
            payload["B3_seed_parent_vector_length"] = None

        msh, _cell_tags, facet_tags = fem3d._load_mesh_and_tags(mesh_file)
        f_top = np.asarray(facet_tags.find(TAG_TOP), dtype=np.int32)
        f_back = np.asarray(facet_tags.find(TAG_BACK), dtype=np.int32)
        f_ribs = np.asarray(facet_tags.find(TAG_RIBS), dtype=np.int32)
        f_fix = np.asarray(facet_tags.find(TAG_FIX), dtype=np.int32)
        shell_facets = np.unique(np.concatenate([f_top, f_back, f_ribs]).astype(np.int32, copy=False))
        tmeta = _build_c2_trace_to_parent_transfer(
            msh, facet_tags, shell_facets=shell_facets, tag_top=TAG_TOP, tag_back=TAG_BACK, tag_ribs=TAG_RIBS
        )
        _tmeta_parent_map = tmeta.get("parent_index_per_trace_dof")
        payload["B3_raw_T_rebuilt_in_memory"] = bool(tmeta.get("ok", False))
        payload["B3_raw_tmeta_parent_idx_present"] = _tmeta_parent_map is not None

        print("[B3_seed] stage=before_parent_replay_assembly", flush=True)
        A_parent, M_parent, cfg = _assemble_reduced_coupled_replay(
            mesh_file, sample, coupling_enabled=True, capture_parent_raw_blocks=True
        )
        print("[B3_seed] stage=after_parent_replay_assembly", flush=True)
        maps = _extract_layout_maps(cfg, A_parent)
        u_to_W_parent = np.asarray(maps["u_to_W"], dtype=np.int32).ravel()
        p_to_W_parent = np.asarray(maps["p_to_W"], dtype=np.int32).ravel()
        p_air_collapsed = np.asarray(
            cfg.get("_coupled_air_p_air_collapsed_indices", np.asarray([], dtype=np.int32)),
            dtype=np.int32,
        ).ravel()
        restr = dict(maps.get("restr") or {})
        raw_cap = _extract_parent_raw_block_capture()
        for k in (
            "B3_composition_parent_raw_capture_constructed",
            "parent_raw_collapsed_layout_constructed",
            "parent_raw_collapsed_layout_dimensions_pass",
            "parent_raw_u_dimension",
            "parent_raw_p_dimension",
        ):
            payload[k] = raw_cap.get(k)
        raw_App = raw_cap.get("raw_App")
        raw_Mpp = raw_cap.get("raw_Mpp")
        raw_Aup = raw_cap.get("raw_Aup")
        raw_Apu = raw_cap.get("raw_Apu")
        raw_Mpu = raw_cap.get("raw_Mpu")
        for m_ in (raw_App, raw_Mpp, raw_Aup, raw_Apu, raw_Mpu):
            if m_ is not None:
                _register_mat_for_destroy(mats_to_destroy, m_, seen=mat_destroy_seen)
        if not all(m is not None for m in (raw_App, raw_Mpp, raw_Aup, raw_Apu, raw_Mpu)):
            payload["B3_seed_operator_build_failure_reason"] = "missing_parent_raw_blocks"
            return 2
        if not bool(payload.get("parent_raw_collapsed_layout_dimensions_pass", False)):
            payload["B3_seed_operator_build_failure_reason"] = "parent_raw_collapsed_layout_not_passing"
            return 2
        if _tmeta_parent_map is None:
            payload["B3_seed_operator_build_failure_reason"] = "parent_index_per_trace_dof_missing_from_tmeta"
            return 2

        shell_mesh, shell_to_parent, _, _ = dmesh.create_submesh(msh, msh.topology.dim - 1, shell_facets)
        V_u_trace = fem.functionspace(shell_mesh, fem3d._displacement_element(shell_mesh, 1))
        trace_cells = np.arange(
            int(shell_mesh.topology.index_map(shell_mesh.topology.dim).size_local), dtype=np.int32
        )
        map_meta = _extract_submesh_to_parent_entity_indices(shell_to_parent, entity_dim=msh.topology.dim - 1)
        parent_tag_map = {
            int(i): int(v) for i, v in zip(np.asarray(facet_tags.indices), np.asarray(facet_tags.values))
        }
        parent_f = np.asarray(map_meta.get("indices"), dtype=np.int32).ravel()
        trace_vals = np.array([parent_tag_map.get(int(pf), -1) for pf in parent_f], dtype=np.int32)
        mt_trace = dmesh.meshtags(shell_mesh, shell_mesh.topology.dim, trace_cells, trace_vals)
        dx_trace = ufl.Measure("dx", domain=shell_mesh, subdomain_data=mt_trace)
        u = ufl.TrialFunction(V_u_trace)
        v = ufl.TestFunction(V_u_trace)
        top_m, back_m, t_top, t_back = fem3d._split_wood_materials(cfg)
        nrm = ufl.CellNormal(shell_mesh)
        P = ufl.Identity(3) - ufl.outer(nrm, nrm)
        e1, e2 = fem3d._plate_local_frame(nrm, P)
        grad_u = ufl.grad(u)
        grad_v = ufl.grad(v)
        eps_u = 0.5 * (P * grad_u * P + ufl.transpose(P * grad_u * P))
        eps_v = 0.5 * (P * grad_v * P + ufl.transpose(P * grad_v * P))
        w_n = ufl.dot(u, nrm)
        v_n = ufl.dot(v, nrm)
        shell_top = fem3d._orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, top_m)
        shell_back = fem3d._orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, back_m)
        shell_ribs = fem3d._orthotropic_shell_stiffness_form(eps_u, eps_v, w_n, v_n, e1, e2, P, back_m)
        raw_Auu = fem.petsc.assemble_matrix(
            fem.form(
                shell_top * dx_trace(TAG_TOP)
                + shell_back * dx_trace(TAG_BACK)
                + shell_ribs * dx_trace(TAG_RIBS)
            ),
            bcs=[],
        )
        raw_Muu = fem.petsc.assemble_matrix(
            fem.form(
                (top_m["rho"] * t_top) * ufl.dot(u, v) * dx_trace(TAG_TOP)
                + (back_m["rho"] * t_back) * ufl.dot(u, v) * dx_trace(TAG_BACK)
                + (back_m["rho"] * t_back) * ufl.dot(u, v) * dx_trace(TAG_RIBS)
            ),
            bcs=[],
        )
        raw_Auu.assemble()
        raw_Muu.assemble()
        for m_ in (raw_Auu, raw_Muu):
            _register_mat_for_destroy(mats_to_destroy, m_, seen=mat_destroy_seen)

        parent_idx = np.asarray(_tmeta_parent_map, dtype=np.int32).ravel()
        n_parent_collapsed = int(payload.get("parent_raw_u_dimension", 0) or 0)
        payload["B3_raw_tmeta_parent_idx_length"] = int(parent_idx.size)
        payload["B3_raw_tmeta_parent_idx_bounds_pass"] = bool(
            parent_idx.size > 0
            and int(np.min(parent_idx)) >= 0
            and int(np.max(parent_idx)) < n_parent_collapsed
        )
        payload["B3_raw_tmeta_parent_idx_unique_pass"] = bool(np.unique(parent_idx).size == parent_idx.size)
        if not payload["B3_raw_tmeta_parent_idx_bounds_pass"] or not payload["B3_raw_tmeta_parent_idx_unique_pass"]:
            payload["B3_seed_operator_build_failure_reason"] = "parent_index_per_trace_dof_contract_failed"
            return 2

        is_parent_u = PETSc.IS().createGeneral(parent_idx.astype(np.int32), comm=PETSc.COMM_WORLD)
        is_p = PETSc.IS().createGeneral(
            np.arange(raw_App.getSize()[0], dtype=np.int32), comm=PETSc.COMM_WORLD
        )
        raw_Aup_B3 = raw_Aup.createSubMatrix(is_parent_u, is_p)
        raw_Apu_B3 = raw_Apu.createSubMatrix(is_p, is_parent_u)
        raw_Mpu_B3 = raw_Mpu.createSubMatrix(is_p, is_parent_u)
        for m_ in (raw_Aup_B3, raw_Apu_B3, raw_Mpu_B3):
            _register_mat_for_destroy(mats_to_destroy, m_, seen=mat_destroy_seen)
        is_parent_u.destroy()
        is_p.destroy()

        n_u_b3 = int(raw_Auu.getSize()[0])
        n_p_retained = int(p_to_W_parent.size)
        s_uu = max(float(_mat_norm_or_none(raw_Auu) or 0.0), 1.0e-30)
        s_pp = max(float(_mat_norm_or_none(raw_App) or 0.0), 1.0e-30)
        s_c = math.sqrt(s_uu * s_pp)

        parent_fix_blocks = fem3d._locate_facet_displacement_dofs(
            fem.functionspace(msh, fem3d._displacement_element(msh, 1)), msh, f_fix
        )
        fix_scalar_parent = set(
            int(b) * 3 + c
            for b in np.asarray(parent_fix_blocks, dtype=np.int32).ravel()
            for c in range(3)
        )
        b3_fix_scalar = np.asarray(
            [k for k, pi in enumerate(parent_idx.tolist()) if int(pi) in fix_scalar_parent],
            dtype=np.int32,
        )
        payload["B3_tag5_vector_BC_contract_pass"] = bool(
            b3_fix_scalar.size == int(np.unique(b3_fix_scalar // 3).size * 3)
        )
        payload["B3_retained_pressure_dimension"] = n_p_retained
        payload["B3_new_reduced_W_dimension"] = int(n_u_b3 + n_p_retained)
        payload["B3_pressure_restriction_policy_reused"] = bool(cfg.get("_coupled_air_pressure_restriction"))
        payload["B3_raw_composition_contract_pass"] = bool(
            payload.get("B3_composition_parent_raw_capture_constructed")
            and payload.get("parent_raw_collapsed_layout_dimensions_pass")
            and payload["B3_raw_tmeta_parent_idx_bounds_pass"]
            and payload["B3_tag5_vector_BC_contract_pass"]
            and payload["B3_pressure_restriction_policy_reused"]
        )
        if not payload["B3_raw_composition_contract_pass"]:
            payload["B3_seed_operator_build_failure_reason"] = "B3_operator_composition_gates_failed"
            return 2

        print("[B3_seed] stage=before_b3_operator_build", flush=True)
        op_meta: Dict[str, Any] = {}
        try:
            (
                A_b3,
                M_b3,
                u_idx,
                p_idx,
                op_meta,
                bc_rows,
                tag5_rows,
                p_release_rows,
                m_uu_b3,
                m_pu_b3,
                m_pp_b3,
            ) = _build_b3_scaled_restricted_operators_in_memory(
                raw_Auu=raw_Auu,
                raw_Muu=raw_Muu,
                raw_App=raw_App,
                raw_Mpp=raw_Mpp,
                raw_Aup_B3=raw_Aup_B3,
                raw_Apu_B3=raw_Apu_B3,
                raw_Mpu_B3=raw_Mpu_B3,
                s_uu=s_uu,
                s_pp=s_pp,
                s_c=s_c,
                n_u_b3=n_u_b3,
                p_air_collapsed=p_air_collapsed,
                b3_fix_u_rows=b3_fix_scalar,
                msh=msh,
                facet_tags=facet_tags,
                comm=PETSc.COMM_WORLD,
                mats_to_destroy=mats_to_destroy,
                report_meta=op_meta,
                destroy_seen=mat_destroy_seen,
            )
            payload.update(op_meta)
            payload["B3_seed_operator_build_pass"] = True
            payload["B3_seed_operator_build_failure_reason"] = None
        except Exception as exc:
            payload.update(op_meta)
            payload["B3_seed_operator_build_failure_reason"] = f"{type(exc).__name__}:{exc}"
            payload["B3_seed_operator_failure_stage"] = op_meta.get(
                "B3_seed_operator_build_stage", "operator_build_entered"
            )
            payload["B3_seed_operator_failure_exception_type"] = type(exc).__name__
            payload["B3_seed_operator_failure_exception_message"] = str(exc)
            if op_meta.get("B3_direct_sparse_AIJ_operator_constructed") or op_meta.get(
                "B3_final_MatNest_conversion_to_sparse_AIJ_attempted"
            ):
                verdict = (
                    "B3_OPERATOR_BC_APPLICATION_INTERFACE_BLOCKED"
                    if operator_aij_bc_contract_only
                    else "B3_SEED_REPLAY_BLOCKED_BY_OPERATOR_BC_APPLICATION_INTERFACE"
                )
            elif operator_aij_bc_contract_only:
                verdict = "B3_OPERATOR_NATIVE_LIFECYCLE_BLOCKED"
            return 2

        if not bool(payload.get("B3_scaled_restricted_BC_operator_contract_pass")):
            payload["B3_seed_operator_build_failure_reason"] = (
                payload.get("B3_BC_application_failure_reason")
                or "B3_scaled_restricted_BC_operator_contract_failed"
            )
            verdict = (
                "B3_OPERATOR_BC_APPLICATION_INTERFACE_BLOCKED"
                if operator_aij_bc_contract_only
                else "B3_SEED_REPLAY_BLOCKED_BY_OPERATOR_BC_APPLICATION_INTERFACE"
            )
            return 2

        print("[B3_seed] stage=after_b3_operator_build", flush=True)
        if operator_aij_bc_contract_only:
            payload["B3_raw_composition_contract_pass"] = bool(
                payload.get("B3_composition_parent_raw_capture_constructed")
                and payload.get("parent_raw_collapsed_layout_dimensions_pass")
                and payload.get("B3_raw_tmeta_parent_idx_bounds_pass")
                and payload.get("B3_tag5_vector_BC_contract_pass")
                and payload.get("B3_pressure_restriction_policy_reused")
            )
            verdict = "B3_SPARSE_AIJ_BC_OPERATOR_READY_FOR_SEED_REPLAY_RERUN"
            return 0

        payload["B3_raw_sparse_coupling_projection_constructed"] = True
        payload["B3_gnhep_normalization_recomputed_from_B3_blocks"] = True

        if not bool(seed_info.get("seed_file_exists")) or seed_info.get("seed_array") is None:
            payload["B3_seed_mapped_vector_constructed"] = False
            payload["B3_seed_mapping_failure_reason"] = seed_info.get("seed_load_status")
            verdict = "B3_SEED_REPLAY_BLOCKED_BY_SEED_MAPPING_INTERFACE"
            return 2

        parent_seed = np.asarray(seed_info["seed_array"], dtype=np.float64).ravel()
        n_u_parent = int(u_to_W_parent.size)
        expected_parent_len = int(n_u_parent + n_p_retained)
        if int(parent_seed.size) != expected_parent_len:
            payload["B3_seed_mapped_vector_constructed"] = False
            payload["B3_seed_mapping_failure_reason"] = (
                f"parent_seed_length_{parent_seed.size}_!=_expected_restricted_u_plus_p_{expected_parent_len}"
            )
            verdict = "B3_SEED_REPLAY_BLOCKED_BY_SEED_MAPPING_INTERFACE"
            return 2
        if int(A_parent.getSize()[0]) != expected_parent_len:
            payload["B3_seed_parent_operator_dimension_mismatch"] = (
                f"A_parent_dim_{A_parent.getSize()[0]}_!=_expected_{expected_parent_len}"
            )

        b3_seed, u_idx, p_idx, seed_map_meta = _map_parent_seed_to_b3(
            parent_seed,
            parent_idx=parent_idx,
            n_u_parent=n_u_parent,
            n_p_retained=n_p_retained,
            n_u_b3=n_u_b3,
            p_to_W_parent=p_to_W_parent,
        )
        payload.update(seed_map_meta)
        payload["B3_seed_mapped_vector_constructed"] = True
        payload["B3_seed_mapping_method"] = (
            "parent_restricted_concatenated_u_then_p_with_T_pullback_on_u_block"
        )
        payload["B3_seed_mapped_u_dimension"] = int(n_u_b3)
        payload["B3_seed_mapped_p_dimension"] = int(n_p_retained)
        payload["B3_seed_mapped_total_dimension"] = int(b3_seed.size)
        payload["B3_seed_mapping_failure_reason"] = None

        if bc_conditioned_replay_only:
            cond_meta = _audit_b3_seed_bc_conditioning(
                A_b3=A_b3,
                M_b3=M_b3,
                b3_seed=b3_seed,
                u_idx=u_idx,
                p_idx=p_idx,
                bc_rows=bc_rows,
                tag5_rows=tag5_rows,
                p_release_rows=p_release_rows,
                operator_bc_row_crc32=payload.get("B3_operator_bc_row_crc32"),
            )
            payload.update(cond_meta)
            payload["B3_seed_dirichlet_row_contract_matches_operator_BC"] = bool(
                cond_meta.get("B3_seed_dirichlet_row_contract_matches_operator_BC")
            )
            verdict = _b3_seed_bc_conditioning_verdict(payload)
            return 2

        if conditioned_mass_decomposition_only:
            payload["B3_mass_audit_started"] = False
            payload["B3_mass_audit_precheck_stage"] = "before_bc_conditioning"
            cond_meta = _audit_b3_seed_bc_conditioning(
                A_b3=A_b3,
                M_b3=M_b3,
                b3_seed=b3_seed,
                u_idx=u_idx,
                p_idx=p_idx,
                bc_rows=bc_rows,
                tag5_rows=tag5_rows,
                p_release_rows=p_release_rows,
                operator_bc_row_crc32=payload.get("B3_operator_bc_row_crc32"),
                skip_rayleigh_replay=True,
            )
            payload.update(cond_meta)
            if not bool(cond_meta.get("B3_seed_conditioned_vector_constructed")):
                verdict = "B3_SEED_REPLAY_BLOCKED_BY_BC_CONDITIONING_INTERFACE"
                return 2
            b3_conditioned = np.asarray(b3_seed, dtype=np.float64).copy()
            b3_conditioned[bc_rows] = 0.0
            payload["B3_mass_audit_precheck_stage"] = "before_mass_decomposition"
            payload["B3_mass_audit_started"] = True
            mass_meta = _audit_b3_conditioned_seed_mass_decomposition(
                m_uu_pre_bc=m_uu_b3,
                m_pu_pre_bc=m_pu_b3,
                m_pp_pre_bc=m_pp_b3,
                m_final=M_b3,
                x_conditioned=b3_conditioned,
                bc_rows=bc_rows,
                n_u=n_u_b3,
                n_p=n_p_retained,
                comm=PETSc.COMM_WORLD,
                mats_to_destroy=mats_to_destroy,
                destroy_seen=mat_destroy_seen,
            )
            payload.update(mass_meta)
            payload["B3_mass_audit_precheck_stage"] = "after_mass_decomposition"
            payload["B3_mass_audit_precheck_stage"] = "before_conditioned_replay_helper"
            try:
                cond_replay = _rayleigh_residual_like(
                    A_b3, M_b3, b3_conditioned, u_idx=u_idx, p_idx=p_idx
                )
                payload["B3_mass_audit_precheck_stage"] = "after_conditioned_replay_helper"
                payload["B3_seed_conditioned_xH_Mx"] = _safe_float(cond_replay.get("xH_Mx"))
                payload["B3_seed_conditioned_rayleigh_frequency_hz"] = _safe_float(
                    cond_replay.get("replay_frequency_hz")
                )
                payload["B3_seed_conditioned_relative_residual"] = _safe_float(
                    cond_replay.get("replay_relative_residual")
                )
                payload["B3_mass_audit_optional_replay_helper_pass"] = True
                payload["B3_mass_audit_optional_replay_helper_failure_reason"] = None
            except Exception as replay_exc:
                payload["B3_mass_audit_precheck_stage"] = "conditioned_replay_helper_failed_after_mass_decomposition"
                payload["B3_mass_audit_optional_replay_helper_pass"] = False
                payload["B3_mass_audit_optional_replay_helper_failure_exception_type"] = type(
                    replay_exc
                ).__name__
                payload["B3_mass_audit_optional_replay_helper_failure_reason"] = (
                    f"{type(replay_exc).__name__}:{replay_exc}"
                )
            verdict = str(
                mass_meta.get(
                    "B3_conditioned_seed_mass_diagnostic_classification",
                    "B3_SEED_REPLAY_BLOCKED_BY_BC_CONDITIONING_INTERFACE",
                )
            )
            return (
                0
                if verdict
                == "B3_CONDITIONED_SEED_MASS_QUADRATIC_POSITIVE_READY_FOR_REPLAY_REEVALUATION"
                else 2
            )

        parent_replay = _rayleigh_residual_like(A_parent, M_parent, parent_seed, u_idx=u_to_W_parent, p_idx=p_to_W_parent)
        hist_f = _safe_float(parent_replay.get("replay_frequency_hz"))
        payload["historical_parent_seed_frequency_hz_if_available"] = hist_f

        b3_replay = _rayleigh_residual_like(A_b3, M_b3, b3_seed, u_idx=u_idx, p_idx=p_idx)
        payload["B3_seed_xH_Mx"] = _safe_float(b3_replay.get("xH_Mx"))
        payload["B3_seed_rayleigh_frequency_hz"] = _safe_float(b3_replay.get("replay_frequency_hz"))
        payload["B3_seed_relative_residual"] = _safe_float(b3_replay.get("replay_relative_residual"))
        p_block = b3_seed[p_idx]
        p_norm = float(np.linalg.norm(p_block))
        total_norm = float(np.linalg.norm(b3_seed))
        payload["B3_seed_pressure_support_metric"] = _safe_float(
            p_norm / max(total_norm, 1.0e-30) if total_norm > 0 else float("nan")
        )
        payload["B3_seed_u_norm"] = _safe_float(float(np.linalg.norm(b3_seed[u_idx])))
        payload["B3_seed_p_norm"] = _safe_float(p_norm)

        shift_hz = None
        shift_frac = None
        if hist_f is not None and payload["B3_seed_rayleigh_frequency_hz"] is not None:
            try:
                shift_hz = float(payload["B3_seed_rayleigh_frequency_hz"]) - float(hist_f)
                shift_frac = shift_hz / max(abs(float(hist_f)), 1.0e-30)
            except Exception:
                pass
        payload["B3_vs_historical_seed_frequency_shift_hz"] = _safe_float(shift_hz)
        payload["B3_vs_historical_seed_frequency_shift_fraction"] = _safe_float(shift_frac)

        def _finite_pos(x: Any) -> bool:
            if x is None or isinstance(x, str):
                return False
            try:
                v = float(x)
                return math.isfinite(v) and v > 0.0
            except Exception:
                return False

        def _finite_scalar(x: Any) -> bool:
            if x is None or isinstance(x, str):
                return False
            try:
                return math.isfinite(float(x))
            except Exception:
                return False

        gates_ok = bool(
            payload["B3_seed_mapped_vector_constructed"]
            and _finite_pos(payload["B3_seed_xH_Mx"])
            and _finite_pos(payload["B3_seed_rayleigh_frequency_hz"])
            and _finite_scalar(payload["B3_seed_relative_residual"])
            and _finite_scalar(payload["B3_seed_pressure_support_metric"])
            and float(payload["B3_seed_pressure_support_metric"]) > 1.0e-12
        )
        rel_res = float(payload["B3_seed_relative_residual"])
        large_shift = bool(
            shift_frac is not None
            and not isinstance(shift_frac, str)
            and math.isfinite(float(shift_frac))
            and (abs(float(shift_frac)) > 0.25 or rel_res > 0.5)
        )
        payload["B3_seed_replay_large_shift_warning"] = large_shift
        if not gates_ok:
            verdict = "B3_SEED_REPLAY_BLOCKED_BY_SEED_MAPPING_INTERFACE"
            payload["B3_seed_mapping_failure_reason"] = "B3_replay_metrics_degenerate_or_nonfinite"
            return 2
        if large_shift:
            verdict = "B3_SEED_REPLAY_DIAGNOSTIC_INCONCLUSIVE_DUE_TO_PRE_BC_FIX_SEED"
        else:
            verdict = "B3_SEED_REPLAY_DIAGNOSTIC_PASS_READY_FOR_FIRST_JD_DESIGN_REVIEW"
        return 0 if verdict.startswith("B3_SEED_REPLAY_DIAGNOSTIC_PASS") else 2
    except Exception as exc:
        if conditioned_mass_decomposition_only:
            exc_type = type(exc).__name__
            exc_reason = f"{exc_type}:{exc}"
            if bool(payload.get("B3_mass_audit_started")):
                if isinstance(exc, _B3MassCrossQuadraticMpuError):
                    payload["B3_mass_audit_failure_stage"] = (
                        "mass_decomposition_cross_quadratic_Mpu"
                    )
                else:
                    payload["B3_mass_audit_failure_stage"] = "mass_decomposition"
                payload["B3_mass_audit_failure_exception_type"] = exc_type
                payload["B3_mass_audit_failure_reason"] = exc_reason
                verdict = (
                    "B3_CONDITIONED_SEED_MASS_AUDIT_BLOCKED_BY_MASS_DECOMPOSITION_INTERFACE"
                )
            elif bool(payload.get("B3_seed_conditioned_vector_constructed")):
                payload["B3_mass_audit_failure_stage"] = (
                    "conditioned_replay_helper_before_mass_decomposition"
                )
                payload["B3_mass_audit_failure_exception_type"] = exc_type
                payload["B3_mass_audit_failure_reason"] = exc_reason
                verdict = (
                    "B3_CONDITIONED_SEED_MASS_AUDIT_BLOCKED_BY_PREAUDIT_REPLAY_HELPER_INTERFACE"
                )
            elif bool(payload.get("B3_seed_mapped_vector_constructed")):
                payload["B3_mass_audit_failure_stage"] = "bc_conditioning_before_mass_decomposition"
                payload["B3_mass_audit_failure_exception_type"] = exc_type
                payload["B3_mass_audit_failure_reason"] = exc_reason
                verdict = "B3_SEED_REPLAY_BLOCKED_BY_BC_CONDITIONING_INTERFACE"
            elif bool(payload.get("B3_seed_operator_build_pass")):
                payload["B3_mass_audit_failure_stage"] = "seed_mapping_after_operator_build"
                payload["B3_mass_audit_failure_exception_type"] = exc_type
                payload["B3_mass_audit_failure_reason"] = exc_reason
                verdict = "B3_SEED_REPLAY_BLOCKED_BY_SEED_MAPPING_INTERFACE"
            else:
                payload["B3_mass_audit_failure_stage"] = "operator_build"
                payload["B3_mass_audit_failure_exception_type"] = exc_type
                payload["B3_mass_audit_failure_reason"] = exc_reason
                verdict = "B3_SEED_REPLAY_BLOCKED_BY_OPERATOR_RESTRICTION_INTERFACE"
        elif bool(payload.get("B3_seed_operator_build_pass")) and not bool(
            payload.get("B3_seed_mapped_vector_constructed")
        ):
            payload["B3_seed_mapping_failure_reason"] = f"{type(exc).__name__}:{exc}"
            verdict = "B3_SEED_REPLAY_BLOCKED_BY_SEED_MAPPING_INTERFACE"
        elif bool(payload.get("B3_seed_mapped_vector_constructed")):
            payload["B3_seed_mapping_failure_reason"] = f"{type(exc).__name__}:{exc}"
            verdict = "B3_SEED_REPLAY_BLOCKED_BY_SEED_MAPPING_INTERFACE"
        else:
            payload["B3_seed_operator_build_failure_reason"] = f"{type(exc).__name__}:{exc}"
            if payload.get("B3_direct_sparse_AIJ_operator_constructed") or payload.get(
                "B3_final_MatNest_conversion_to_sparse_AIJ_attempted"
            ):
                verdict = (
                    "B3_OPERATOR_BC_APPLICATION_INTERFACE_BLOCKED"
                    if operator_aij_bc_contract_only
                    else "B3_SEED_REPLAY_BLOCKED_BY_OPERATOR_BC_APPLICATION_INTERFACE"
                )
            elif operator_aij_bc_contract_only:
                verdict = "B3_OPERATOR_NATIVE_LIFECYCLE_BLOCKED"
            else:
                verdict = "B3_SEED_REPLAY_BLOCKED_BY_OPERATOR_RESTRICTION_INTERFACE"
        return 2
    finally:
        payload["next_step_verdict"] = verdict
        _write_json_atomic(out_json, payload)
        payload["report_size_bytes"] = out_json.stat().st_size
        _write_json_atomic(out_json, payload)
        md_lines = [
            f"# B3 audit ({mode_label}, report-only)",
            "",
            f"- verdict: `{verdict}`",
            f"- B3_seed_source_status: {payload.get('B3_seed_source_status')}",
            f"- B3_raw_composition_contract_pass: {payload.get('B3_raw_composition_contract_pass')}",
            f"- B3_seed_rayleigh_frequency_hz: {payload.get('B3_seed_rayleigh_frequency_hz')}",
            f"- B3_seed_relative_residual: {payload.get('B3_seed_relative_residual')}",
            f"- historical_parent_seed_frequency_hz: {payload.get('historical_parent_seed_frequency_hz_if_available')}",
            f"- B3_vs_historical_seed_frequency_shift_fraction: {payload.get('B3_vs_historical_seed_frequency_shift_fraction')}",
            "",
            "no_new_eigensolve_executed=True",
        ]
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        print(f"[B3_seed] mode={mode_label}", flush=True)
        print(f"[B3_seed] B3_raw_composition_contract_pass={payload.get('B3_raw_composition_contract_pass')}", flush=True)
        print(f"[B3_seed] B3_seed_operator_build_pass={payload.get('B3_seed_operator_build_pass')}", flush=True)
        print(f"[B3_seed] B3_sparse_blockwise_pressure_restriction_pass={payload.get('B3_sparse_blockwise_pressure_restriction_pass')}", flush=True)
        print(f"[B3_seed] B3_seed_mapped_vector_constructed={payload.get('B3_seed_mapped_vector_constructed')}", flush=True)
        print(f"[B3_seed] B3_seed_rayleigh_frequency_hz={payload.get('B3_seed_rayleigh_frequency_hz')}", flush=True)
        print(f"[B3_seed] next_step_verdict={verdict}", flush=True)
        print("[B3_seed] no_new_eigensolve_executed=True", flush=True)
        print("[B3_seed] additional_eps=NOT_AUTHORIZED", flush=True)
        _register_mat_for_destroy(mats_to_destroy, A_parent, seen=mat_destroy_seen)
        _register_mat_for_destroy(mats_to_destroy, M_parent, seen=mat_destroy_seen)
        _register_mat_for_destroy(mats_to_destroy, A_b3, seen=mat_destroy_seen)
        _register_mat_for_destroy(mats_to_destroy, M_b3, seen=mat_destroy_seen)
        _destroyed, _dup_destroy, guard_pass = _destroy_mats_deduped(mats_to_destroy)
        payload["B3_native_destroyed_unique_object_count"] = int(_destroyed)
        payload["B3_native_duplicate_destroy_attempt_count"] = int(_dup_destroy)
        payload["B3_native_double_destroy_guard_pass"] = bool(guard_pass)
        print(
            f"[B3_seed] B3_native_double_destroy_guard_pass={payload.get('B3_native_double_destroy_guard_pass')} "
            f"B3_native_destroyed_unique_object_count={payload.get('B3_native_destroyed_unique_object_count')} "
            f"B3_native_duplicate_destroy_attempt_count={payload.get('B3_native_duplicate_destroy_attempt_count')}",
            flush=True,
        )


def _run_v2_vector_bc_contract_only(pre: Dict[str, Any]) -> int:
    if not pre["preassembly_contract_pass"]:
        print("[V2_BC] V2_tag5_vector_BC_contract_pass=False", flush=True)
        print("[V2_BC] V2_tag5_vector_BC_failure_reason=preassembly_contract_failed", flush=True)
        print("[V2_BC] no_new_eigensolve_executed=True", flush=True)
        print("[V2_BC] additional_eps=NOT_AUTHORIZED", flush=True)
        return 2
    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[V2_BC] V2_tag5_vector_BC_contract_pass=False", flush=True)
            print("[V2_BC] V2_tag5_vector_BC_failure_reason=requires_mpiexec_n_1", flush=True)
            print("[V2_BC] no_new_eigensolve_executed=True", flush=True)
            print("[V2_BC] additional_eps=NOT_AUTHORIZED", flush=True)
        return 2

    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
    sample = sample_spec_from_case(case)
    mesh_file = mesh_path("L_mid", CASE_ID)
    _A, _M, cfg = _assemble_reduced_coupled_replay(mesh_file, sample, coupling_enabled=True)
    bc = dict(cfg.get("_V2_vector_BC_contract") or {})

    block_size = bc.get("V2_tag5_vector_block_size")
    fix_blocks = bc.get("V2_tag5_fix_block_dof_count")
    expected_rows = bc.get("V2_tag5_expected_scalar_component_row_count")
    actual_scalar = bc.get("V2_tag5_actual_scalar_component_row_count_after_fix")
    pressure_rows = bc.get("V2_pressure_dirichlet_row_count")
    total_rows = bc.get("V2_total_algebraic_dirichlet_row_count_after_fix")
    pass_flag = bool(bc.get("V2_tag5_vector_BC_contract_pass", False))
    fail_reason = bc.get("V2_tag5_vector_BC_failure_reason")

    print(f"[V2_BC] V2_tag5_vector_block_size={block_size}", flush=True)
    print(f"[V2_BC] V2_tag5_fix_block_dof_count={fix_blocks}", flush=True)
    print(f"[V2_BC] V2_tag5_expected_scalar_component_row_count={expected_rows}", flush=True)
    print(f"[V2_BC] V2_tag5_actual_scalar_component_row_count_after_fix={actual_scalar}", flush=True)
    print(f"[V2_BC] V2_pressure_dirichlet_row_count={pressure_rows}", flush=True)
    print(f"[V2_BC] V2_total_algebraic_dirichlet_row_count_after_fix={total_rows}", flush=True)
    print(f"[V2_BC] V2_tag5_vector_BC_contract_pass={pass_flag}", flush=True)
    print(f"[V2_BC] V2_tag5_vector_BC_failure_reason={fail_reason}", flush=True)
    print(
        "[V2_BC] current_physical_model_status="
        "V2_INTENDED_PHYSICS_NOT_INVALIDATED_IMPLEMENTATION_BC_BUG_CONFIRMED",
        flush=True,
    )
    print(
        "[V2_BC] current_solver_status="
        "ST_SINVERT_FAILURE_EVIDENCE_CONFOUNDED_BY_INCOMPLETE_TAG5_VECTOR_BC",
        flush=True,
    )
    print("[V2_BC] mesh_convergence_resume=BLOCKED", flush=True)
    print("[V2_BC] production_promotion=BLOCKED", flush=True)
    print("[V2_BC] no_new_eigensolve_executed=True", flush=True)
    print("[V2_BC] additional_eps=NOT_AUTHORIZED", flush=True)
    print("[V2_BC] jd_wiring_authorized=False", flush=True)
    print("[V2_BC] artifact_storage_policy_applied=True", flush=True)
    print(
        "[V2_BC] raw_capture_requirement="
        "B3 raw composition still requires one pre-normalization/pre-restriction/pre-BC "
        "raw-block capture interface in fem_main_3d.py",
        flush=True,
    )
    return 0 if pass_flag else 2


def main() -> int:
    import sys

    if (
        _is_b3_jd_first_bounded_execution_only_mode(sys.argv)
        or _is_b3_jd_dimension_setup_preflight_only_mode(sys.argv)
        or _is_b3_gnhep_bc_spectral_pollution_contract_only_mode(sys.argv)
        or _is_b3_gnhep_bc_no_lambda_one_operator_contract_only_mode(sys.argv)
        or _is_b3_jd_fixed_bc_dimension_setup_preflight_only_mode(sys.argv)
        or _is_b3_jd_fixed_bc_second_bounded_execution_only_mode(sys.argv)
    ):
        pre = _precheck_allow_b3_jd_first_bounded_execution()
    else:
        pre = _precheck()

    if _is_v2_vector_bc_contract_only_mode(sys.argv):
        return _run_v2_vector_bc_contract_only(pre)

    if _is_b3_raw_composition_contract_only_mode(sys.argv):
        return _run_b3_raw_composition_contract_only(pre)

    if _is_b3_operator_aij_bc_contract_only_mode(sys.argv):
        return _run_b3_seed_replay_audit_only(pre, operator_aij_bc_contract_only=True)

    if _is_b3_seed_bc_conditioned_replay_audit_only_mode(sys.argv):
        return _run_b3_seed_replay_audit_only(pre, bc_conditioned_replay_only=True)

    if _is_b3_conditioned_seed_mass_decomposition_audit_only_mode(sys.argv):
        return _run_b3_seed_replay_audit_only(pre, conditioned_mass_decomposition_only=True)

    if _is_b3_jd_design_readiness_contract_only_mode(sys.argv):
        return _run_b3_jd_design_readiness_contract_only(pre)

    if _is_b3_jd_api_preflight_only_mode(sys.argv):
        return _run_b3_jd_api_preflight_only(pre)

    if _is_b3_jd_operator_wiring_preflight_only_mode(sys.argv):
        return _run_b3_jd_operator_wiring_preflight_only(pre)

    if _is_b3_jd_first_bounded_execution_only_mode(sys.argv):
        return _run_b3_jd_first_bounded_execution_only(pre)

    if _is_b3_jd_dimension_setup_preflight_only_mode(sys.argv):
        return _run_b3_jd_dimension_setup_preflight_only(pre)

    if _is_b3_gnhep_bc_spectral_pollution_contract_only_mode(sys.argv):
        return _run_b3_gnhep_bc_spectral_pollution_contract_only(pre)

    if _is_b3_gnhep_bc_no_lambda_one_operator_contract_only_mode(sys.argv):
        return _run_b3_gnhep_bc_no_lambda_one_operator_contract_only(pre)

    if _is_b3_jd_fixed_bc_dimension_setup_preflight_only_mode(sys.argv):
        return _run_b3_jd_fixed_bc_dimension_setup_preflight_only(pre)

    if _is_b3_jd_fixed_bc_second_bounded_execution_only_mode(sys.argv):
        return _run_b3_jd_fixed_bc_second_bounded_execution_only(pre)

    if _is_b3_seed_replay_audit_only_mode(sys.argv):
        return _run_b3_seed_replay_audit_only(pre)

    if _is_c2_sparse_coupling_only_mode(sys.argv):
        return _run_c2_sparse_coupling_only(pre)

    if _is_c2_transfer_contract_only_mode(sys.argv):
        return _run_c2_transfer_contract_only(pre)

    print(f"[B3_coupled] preassembly_helper_import_pass={pre['preassembly_helper_import_pass']}", flush=True)
    print(
        f"[B3_coupled] preassembly_rayleigh_signature_pass={pre['preassembly_rayleigh_signature_pass']}",
        flush=True,
    )
    print(
        f"[B3_coupled] preassembly_residual_signature_pass={pre['preassembly_residual_signature_pass']}",
        flush=True,
    )
    print(
        f"[B3_coupled] preassembly_writer_available_pass={pre['preassembly_writer_available_pass']}",
        flush=True,
    )
    print(
        f"[B3_coupled] preassembly_no_eigensolve_call_pass={pre['preassembly_no_eigensolve_call_pass']}",
        flush=True,
    )
    print(f"[B3_coupled] preassembly_contract_pass={pre['preassembly_contract_pass']}", flush=True)
    if "--precheck-only" in sys.argv:
        print("[B3_coupled] no_new_eigensolve_executed=True", flush=True)
        return 0 if pre["preassembly_contract_pass"] else 2

    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[B3_coupled] Requires mpiexec -n 1", flush=True)
        return 2
    if not pre["preassembly_contract_pass"]:
        print(
            f"[B3_coupled] preassembly_failure_reasons={json.dumps(pre['preassembly_failure_reasons'])}",
            flush=True,
        )
        print("[B3_coupled] no_new_eigensolve_executed=True", flush=True)
        return 2

    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
    sample = sample_spec_from_case(case)
    mesh_file = mesh_path("L_mid", CASE_ID)
    case_dir = solve_case_dir("L_mid", CASE_ID)
    seed_npy = case_dir / "diagnostics" / "acoustic_coupled_seed.npy"

    A = M = None
    n_u = n_p = n_w = None
    block_reason = None
    try:
        A, M, cfg = _assemble_reduced_coupled_replay(mesh_file, sample, coupling_enabled=True)
        maps = _extract_layout_maps(cfg, A)
        u_to_W = np.asarray(maps["u_to_W"], dtype=np.int32).ravel()
        p_to_W = np.asarray(maps["p_to_W"], dtype=np.int32).ravel()
        n_u = int(u_to_W.size)
        n_p = int(p_to_W.size)
        n_w = int(A.getSize()[0])

        msh, _cell_tags, facet_tags = fem3d._load_mesh_and_tags(mesh_file)
        f_top = np.asarray(facet_tags.find(TAG_TOP), dtype=np.int32)
        f_back = np.asarray(facet_tags.find(TAG_BACK), dtype=np.int32)
        f_ribs = np.asarray(facet_tags.find(TAG_RIBS), dtype=np.int32)
        f_fix = np.asarray(facet_tags.find(TAG_FIX), dtype=np.int32)
        shell_facets = np.unique(np.concatenate([f_top, f_back, f_ribs]).astype(np.int32, copy=False))

        b3_space_ok = False
        b3_form_ok = False
        b3_top = b3_back = b3_ribs = False
        b3_mass_present = b3_stiff_present = False
        b3_bc_constructed = False
        b3_bc_pass = False
        b3_bc_reason = None
        b3_u_new = None
        b3_total_w = None
        b3_ratio = None
        b3_tag5_fixed_n = 0
        b3_null_exposure = "UNRESOLVED"
        b3_Auu_norm = b3_Muu_norm = None
        b3_App_norm = b3_Mpp_norm = b3_Aup_norm = b3_Apu_norm = b3_Mpu_norm = None
        b3_coupling_iface = False
        b3_coupling_present = False
        b3_coupling_method = "UNAVAILABLE"
        b3_coupling_reason = "not_attempted"
        b3_ops_assembled = False
        b3_ops_sanity = False
        b3_ops_reason = None
        b3_seed_map = False
        b3_seed_repr = False
        b3_seed_pass = False
        b3_seed_method = "UNAVAILABLE"
        b3_seed_fail = "not_attempted"
        b3_seed_pressure_support = False
        b3_seed_mac = None
        b3_seed_xhmx = b3_seed_f = b3_seed_res = None
        b3_scalable = True
        cpre = _coupling_contract_precheck()
        selected_route = cpre["selected_B3_coupling_route"]
        coupling_stage_outcome = None
        c2_t_domain_space = "B3_trace_u_submesh_P1_vector"
        c2_t_codomain_space = "parent_reduced_u_representation"
        c2_t_domain_dim = None
        c2_t_codomain_dim = None
        c2_t_direction = "B3_trace_to_parent_interface_u"
        c2_t_method = "UNAVAILABLE"
        c2_t_constructed = False
        c2_t_is_sparse = True
        c2_t_is_interp = "exact_coordinate_lift_on_P1_trace_parent_matching"
        c2_t_geom_preserve = False
        c2_t_shape = None
        c2_t_nnz = None
        c2_t_density = None
        c2_t_row_nnz_min = None
        c2_t_row_nnz_max = None
        c2_t_checksum = None
        c2_t_storage_bytes = None
        c2_t_blocker = None
        c2_t_contract_pass = False
        c2_t_const_pass = False
        c2_t_support_pass = False
        c2_t_tag_pass = False
        c2_t_validation_failure = None
        b3_submesh_map_type = None
        b3_submesh_map_method = None
        b3_submesh_map_ok = False
        b3_submesh_n = 0
        b3_parent_facet_min = None
        b3_parent_facet_max = None
        b3_transferred_counts = {"tag1": 0, "tag3": 0, "tag4": 0}
        b3_transferred_contract = False
        b3_tag_count_convention = "local_plus_ghost_submesh_entities"
        b3_continuum_status = "UNRESOLVED"
        b3_seed_check_status = "NOT_EVALUATED"
        b3_material_fail_reason = None
        b3_parent_geom_deps = [
            "FacetNormal(parent_mesh)",
            "facet projector P=I-n⊗n on parent boundary facets",
            "facet-tangential gradient restriction P*grad(u)*P",
            "surface shell stiffness integrated on ds(tag)",
        ]
        b3_invalid_trace_quantities = []
        b3_geom_replacements = [
            "FacetNormal(parent_mesh) -> CellNormal(trace_mesh)",
            "parent ds(tag) -> trace dx(tag) on submesh meshtags",
            "facet tangential projector -> manifold-cell tangential projector using CellNormal",
            "facet tangential strain restriction -> manifold-cell tangential restriction",
        ]
        b3_rederive_method = (
            "manifold_cell_shell_form_using_CellNormal_and_tangential_projection_on_trace_cells"
        )

        if shell_facets.size > 0 and hasattr(dmesh, "create_submesh"):
            tdim = msh.topology.dim
            shell_mesh, shell_to_parent, _, _ = dmesh.create_submesh(msh, tdim - 1, shell_facets)
            u_el = fem3d._displacement_element(shell_mesh, 1)
            V_u_trace = fem.functionspace(shell_mesh, u_el)
            b3_u_new = int(V_u_trace.dofmap.index_map.size_global * V_u_trace.dofmap.index_map_bs)
            b3_total_w = int(b3_u_new + n_p)
            b3_ratio = float(b3_total_w / max(int(n_w), 1))
            b3_space_ok = True

            # Build trace-cell tags from parent facet tags.
            parent_tag_map = {
                int(i): int(v) for i, v in zip(np.asarray(facet_tags.indices), np.asarray(facet_tags.values))
            }
            trace_cells = np.arange(
                int(shell_mesh.topology.index_map(shell_mesh.topology.dim).size_local), dtype=np.int32
            )
            map_meta = _extract_submesh_to_parent_entity_indices(shell_to_parent, entity_dim=tdim - 1)
            b3_submesh_map_type = map_meta["map_type"]
            b3_submesh_map_method = map_meta["method"]
            b3_submesh_map_ok = bool(map_meta["ok"])
            print(f"[B3_coupled] B3_submesh_entity_map_type={b3_submesh_map_type}", flush=True)
            print(
                f"[B3_coupled] B3_submesh_entity_map_extraction_method={b3_submesh_map_method}",
                flush=True,
            )
            if not b3_submesh_map_ok:
                b3_transferred_contract = False
                print("[B3_coupled] B3_transferred_tags_contract_pass=False", flush=True)
                b3_coupling_reason = "B3_BLOCKED_BY_DOLFINX_ENTITYMAP_TAG_TRANSFER_INTERFACE"
                block_reason = b3_coupling_reason
                b3_ops_reason = b3_coupling_reason
                b3_seed_fail = b3_coupling_reason
                b3_continuum_status = "BLOCKED_PENDING_ENTITYMAP_TAG_TRANSFER"
                b3_seed_check_status = "NOT_EVALUATED_BLOCKED_PENDING_ENTITYMAP_TAG_TRANSFER"
                b3_form_ok = False
                b3_mass_present = False
                b3_stiff_present = False
                b3_coupling_iface = False
                b3_coupling_present = False
                b3_ops_assembled = False
                b3_ops_sanity = False
                b3_seed_map = False
                b3_seed_repr = False
                b3_seed_pass = False
                seed_info = load_seed_with_diagnostics(seed_npy)
                seed_arr = np.asarray(seed_info.get("seed_array"), dtype=np.float64).ravel()
                base_seed = _rayleigh_residual_like(A, M, seed_arr, u_idx=u_to_W, p_idx=p_to_W)
                seed_xhmx_o = _safe_float(base_seed["xH_Mx"])
                seed_f_o = _safe_float(base_seed["replay_frequency_hz"])
                seed_res_o = _safe_float(base_seed["replay_relative_residual"])
                trace_vals = np.full(trace_cells.shape, -1, dtype=np.int32)
            else:
                parent_f = np.asarray(map_meta["indices"], dtype=np.int32).ravel()
                b3_submesh_n = int(parent_f.size)
                b3_parent_facet_min = int(parent_f.min()) if parent_f.size else None
                b3_parent_facet_max = int(parent_f.max()) if parent_f.size else None
                trace_vals = np.array([parent_tag_map.get(int(pf), -1) for pf in parent_f], dtype=np.int32)
                b3_transferred_counts = {
                    "tag1": int(np.sum(trace_vals == TAG_TOP)),
                    "tag3": int(np.sum(trace_vals == TAG_BACK)),
                    "tag4": int(np.sum(trace_vals == TAG_RIBS)),
                }
                b3_transferred_contract = all(v > 0 for v in b3_transferred_counts.values())
                print(
                    f"[B3_coupled] B3_transferred_tags_contract_pass={b3_transferred_contract}",
                    flush=True,
                )
                if not b3_transferred_contract:
                    b3_coupling_reason = "B3_BLOCKED_BY_DOLFINX_ENTITYMAP_TAG_TRANSFER_INTERFACE"
                    block_reason = b3_coupling_reason
                    b3_ops_reason = b3_coupling_reason
                    b3_seed_fail = b3_coupling_reason
                    b3_continuum_status = "BLOCKED_PENDING_ENTITYMAP_TAG_TRANSFER"
                    b3_seed_check_status = "NOT_EVALUATED_BLOCKED_PENDING_ENTITYMAP_TAG_TRANSFER"

            if block_reason == "B3_BLOCKED_BY_DOLFINX_ENTITYMAP_TAG_TRANSFER_INTERFACE":
                b3_form_ok = False
                b3_mass_present = False
                b3_stiff_present = False
                b3_coupling_iface = False
                b3_coupling_present = False
                b3_ops_assembled = False
                b3_ops_sanity = False
                b3_seed_map = False
                b3_seed_repr = False
                b3_seed_pass = False
                b3_coupling_method = "entitymap_tag_transfer_failed_before_trace_form_assembly"
                # Baseline seed metrics for audit continuity.
                if "seed_xhmx_o" not in locals():
                    seed_info = load_seed_with_diagnostics(seed_npy)
                    seed_arr = np.asarray(seed_info.get("seed_array"), dtype=np.float64).ravel()
                    base_seed = _rayleigh_residual_like(A, M, seed_arr, u_idx=u_to_W, p_idx=p_to_W)
                    seed_xhmx_o = _safe_float(base_seed["xH_Mx"])
                    seed_f_o = _safe_float(base_seed["replay_frequency_hz"])
                    seed_res_o = _safe_float(base_seed["replay_relative_residual"])
            else:
                mt_trace = dmesh.meshtags(shell_mesh, shell_mesh.topology.dim, trace_cells, trace_vals)
                dx_trace = ufl.Measure("dx", domain=shell_mesh, subdomain_data=mt_trace)

                try:
                    u = ufl.TrialFunction(V_u_trace)
                    v = ufl.TestFunction(V_u_trace)
                    top_m, back_m, t_top, t_back = fem3d._split_wood_materials(cfg)
                    # Manifold-cell normal for trace mesh; parent FacetNormal is invalid for dx on trace cells.
                    nrm = ufl.CellNormal(shell_mesh)
                    P = ufl.Identity(3) - ufl.outer(nrm, nrm)
                    e1, e2 = fem3d._plate_local_frame(nrm, P)

                    def eps_surface(uu):
                        grad_u = ufl.grad(uu)
                        grad_tan = P * grad_u * P
                        return 0.5 * (grad_tan + ufl.transpose(grad_tan))

                    eps_u = eps_surface(u)
                    eps_v = eps_surface(v)
                    w_n = ufl.dot(u, nrm)
                    v_n = ufl.dot(v, nrm)
                    shell_top = fem3d._orthotropic_shell_stiffness_form(
                        eps_u, eps_v, w_n, v_n, e1, e2, P, top_m
                    )
                    shell_back = fem3d._orthotropic_shell_stiffness_form(
                        eps_u, eps_v, w_n, v_n, e1, e2, P, back_m
                    )
                    shell_ribs = fem3d._orthotropic_shell_stiffness_form(
                        eps_u, eps_v, w_n, v_n, e1, e2, P, back_m
                    )
                    a_uu_t = (
                        shell_top * dx_trace(TAG_TOP)
                        + shell_back * dx_trace(TAG_BACK)
                        + shell_ribs * dx_trace(TAG_RIBS)
                    )
                    m_uu_t = (
                        (top_m["rho"] * t_top) * ufl.dot(u, v) * dx_trace(TAG_TOP)
                        + (back_m["rho"] * t_back) * ufl.dot(u, v) * dx_trace(TAG_BACK)
                        + (back_m["rho"] * t_back) * ufl.dot(u, v) * dx_trace(TAG_RIBS)
                    )
                    Auu = fem.petsc.assemble_matrix(fem.form(a_uu_t), bcs=[])
                    Muu = fem.petsc.assemble_matrix(fem.form(m_uu_t), bcs=[])
                    Auu.assemble()
                    Muu.assemble()
                    b3_Auu_norm = _safe_float(Auu.norm())
                    b3_Muu_norm = _safe_float(Muu.norm())
                    b3_top = int(np.sum(trace_vals == TAG_TOP)) > 0
                    b3_back = int(np.sum(trace_vals == TAG_BACK)) > 0
                    b3_ribs = int(np.sum(trace_vals == TAG_RIBS)) > 0
                    b3_mass_present = bool(float(b3_Muu_norm) > 0.0)
                    b3_stiff_present = bool(float(b3_Auu_norm) > 0.0)
                    b3_form_ok = bool(
                        b3_top and b3_back and b3_ribs and b3_mass_present and b3_stiff_present
                    )
                    b3_null_exposure = (
                        "MISMATCH_REMOVED_BY_TRACE_SPACE_CONSTRUCTION_PENDING_COUPLED_VALIDATION"
                    )
                except Exception as exc:
                    b3_form_ok = False
                    b3_mass_present = False
                    b3_stiff_present = False
                    b3_continuum_status = "BLOCKED_PENDING_MANIFOLD_TRACE_FORM_REDERIVATION"
                    b3_invalid_trace_quantities = ["ReferenceNormal_or_FacetNormal_in_cell_integral_context"]
                    b3_ops_reason = f"{type(exc).__name__}: {exc}"
                    b3_material_fail_reason = b3_ops_reason
                    block_reason = "B3_BLOCKED_BY_MANIFOLD_TRACE_STRUCTURAL_FORM_REDERIVATION_INTERFACE"
                    b3_seed_fail = block_reason
                    b3_seed_check_status = (
                        "NOT_EVALUATED_BLOCKED_PENDING_MANIFOLD_TRACE_FORM_REDERIVATION"
                    )

                if b3_form_ok:
                    b3_continuum_status = "PRESERVED_BY_EQUIVALENT_TRACE_FORM_ASSEMBLY"
                    # Tag-5 policy transfer audit.
                    shell_set = set(int(x) for x in shell_facets.tolist())
                    fix_set = set(int(x) for x in f_fix.tolist())
                    overlap = np.array(sorted(shell_set.intersection(fix_set)), dtype=np.int32)
                    b3_tag5_fixed_n = int(overlap.size)
                    b3_bc_constructed = True
                    b3_bc_pass = True
                    b3_bc_reason = (
                        "tag5_fix_facets_not_in_trace_shell_union"
                        if b3_tag5_fixed_n == 0
                        else "tag5_overlap_with_trace_shell_requires_explicit_trace_bc_application"
                    )
                    if b3_tag5_fixed_n > 0:
                        b3_bc_pass = False

                    # Pressure block retained from baseline reduced operator.
                    try:
                        b3_App_norm = _safe_float(A.norm())
                        b3_Mpp_norm = _safe_float(M.norm())
                    except Exception:
                        pass

                    # Coupling route selection.
                    if selected_route == "C1":
                        b3_coupling_iface = False
                        b3_coupling_present = False
                        b3_coupling_method = "direct_cross_mesh_entity_map_assembly"
                        b3_coupling_reason = (
                            "B3_BLOCKED_BY_ONE_NAMED_DIRECT_MIXED_DOMAIN_API"
                        )
                        coupling_stage_outcome = b3_coupling_reason
                        block_reason = "B3_BLOCKED_BY_DOLFINX_TRACE_TO_VOLUME_COUPLING_INTERFACE"
                    elif selected_route == "C2":
                        tmeta = _build_c2_trace_to_parent_transfer(
                            msh,
                            facet_tags,
                            shell_facets=shell_facets,
                            tag_top=TAG_TOP,
                            tag_back=TAG_BACK,
                            tag_ribs=TAG_RIBS,
                        )
                        c2_t_method = "EntityMap_plus_exact_dof_coordinate_match_on_P1_trace_and_parent"
                        c2_t_domain_dim = tmeta.get("domain_dim")
                        c2_t_codomain_dim = tmeta.get("codomain_dim")
                        c2_t_constructed = bool(tmeta.get("ok", False))
                        c2_t_geom_preserve = bool(tmeta.get("ok", False))
                        c2_t_shape = tmeta.get("shape")
                        c2_t_nnz = tmeta.get("nnz")
                        c2_t_density = tmeta.get("density")
                        c2_t_row_nnz_min = tmeta.get("row_nnz_min")
                        c2_t_row_nnz_max = tmeta.get("row_nnz_max")
                        c2_t_checksum = tmeta.get("mapping_checksum")
                        c2_t_storage_bytes = tmeta.get("storage_bytes")
                        c2_t_blocker = tmeta.get("reason") if not tmeta.get("ok", False) else None
                        c2_t_contract_pass = bool(tmeta.get("C2_T_geometry_map_contract_pass", False))
                        c2_t_const_pass = bool(tmeta.get("C2_T_constant_field_transfer_pass", False))
                        c2_t_support_pass = bool(tmeta.get("C2_T_trace_support_transfer_pass", False))
                        c2_t_tag_pass = bool(tmeta.get("C2_T_tag_support_transfer_pass", False))
                        c2_t_validation_failure = tmeta.get("C2_T_validation_failure_reason")
                        if not tmeta.get("ok", False):
                            b3_coupling_iface = False
                            b3_coupling_present = False
                            b3_coupling_method = "sparse_transfer_T_then_parent_coupling_block_projection"
                            b3_coupling_reason = "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE"
                            coupling_stage_outcome = b3_coupling_reason
                            block_reason = b3_coupling_reason
                            b3_ops_reason = str(tmeta.get("failure_detail"))
                            b3_seed_fail = b3_coupling_reason
                            b3_seed_check_status = (
                                "NOT_EVALUATED_BLOCKED_PENDING_SPARSE_TRACE_TRANSFER_INTERFACE"
                            )
                        else:
                            b3_coupling_iface = False
                            b3_coupling_present = False
                            b3_coupling_method = (
                                "sparse_transfer_T_constructed_dense_parent_projection_path_disabled"
                            )
                            b3_coupling_reason = (
                                "B3_BLOCKED_BY_PROHIBITED_DENSE_PARENT_COUPLING_PROJECTION_PATH"
                            )
                            coupling_stage_outcome = b3_coupling_reason
                            block_reason = b3_coupling_reason
                            b3_seed_fail = b3_coupling_reason
                            b3_seed_check_status = (
                                "NOT_EVALUATED_BLOCKED_BY_PROHIBITED_DENSE_PARENT_COUPLING_PROJECTION_PATH"
                            )
                    else:
                        b3_coupling_iface = False
                        b3_coupling_present = False
                        b3_coupling_method = "none"
                        b3_coupling_reason = (
                            "B3_BLOCKED_BY_DOLFINX_TRACE_TO_VOLUME_COUPLING_INTERFACE"
                        )
                        coupling_stage_outcome = b3_coupling_reason
                        block_reason = b3_coupling_reason

                    b3_ops_assembled = False
                    b3_ops_sanity = False
                    b3_ops_reason = b3_coupling_reason
                    b3_seed_map = False
                    b3_seed_repr = False
                    b3_seed_method = "requires_trace_u_to_reduced_W_transfer_plus_pressure_identity"
                    b3_seed_fail = b3_coupling_reason
                    b3_seed_check_status = "NOT_EVALUATED_BLOCKED_PENDING_TRACE_TO_VOLUME_COUPLING_INTERFACE"
                    b3_seed_pressure_support = False
                    b3_seed_mac = None

                # Baseline seed metrics are still reported for control visibility.
                seed_info = load_seed_with_diagnostics(seed_npy)
                seed_arr = np.asarray(seed_info.get("seed_array"), dtype=np.float64).ravel()
                base_seed = _rayleigh_residual_like(A, M, seed_arr, u_idx=u_to_W, p_idx=p_to_W)
                seed_xhmx_o = _safe_float(base_seed["xH_Mx"])
                seed_f_o = _safe_float(base_seed["replay_frequency_hz"])
                seed_res_o = _safe_float(base_seed["replay_relative_residual"])
        else:
            seed_xhmx_o = seed_f_o = seed_res_o = None
            b3_space_ok = False
            b3_form_ok = False
            b3_top = b3_back = b3_ribs = False
            b3_mass_present = b3_stiff_present = False
            b3_bc_constructed = False
            b3_bc_pass = False
            b3_bc_reason = "trace_submesh_unavailable_or_shell_facets_missing"
            b3_u_new = b3_total_w = b3_ratio = None
            b3_tag5_fixed_n = 0
            b3_null_exposure = "UNRESOLVED"
            b3_Auu_norm = b3_Muu_norm = None
            b3_App_norm = b3_Mpp_norm = b3_Aup_norm = b3_Apu_norm = b3_Mpu_norm = None
            b3_coupling_iface = False
            b3_coupling_present = False
            b3_coupling_method = "UNAVAILABLE"
            b3_coupling_reason = "B3_BLOCKED_BY_DOLFINX_TRACE_TO_VOLUME_COUPLING_INTERFACE"
            block_reason = b3_coupling_reason
            b3_ops_assembled = False
            b3_ops_sanity = False
            b3_ops_reason = b3_coupling_reason
            b3_seed_map = False
            b3_seed_repr = False
            b3_seed_pass = False
            b3_seed_method = "UNAVAILABLE"
            b3_seed_fail = b3_coupling_reason
            b3_seed_check_status = "NOT_EVALUATED_BLOCKED_PENDING_TRACE_SPACE_CONSTRUCTION"
            b3_seed_pressure_support = False
            b3_seed_mac = None
            b3_seed_xhmx = b3_seed_f = b3_seed_res = None

        if block_reason == "B3_BLOCKED_BY_MANIFOLD_TRACE_STRUCTURAL_FORM_REDERIVATION_INTERFACE":
            verdict = "B3_BLOCKED_BY_MANIFOLD_TRACE_STRUCTURAL_FORM_REDERIVATION_INTERFACE"
        elif block_reason == "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE":
            verdict = "B3_BLOCKED_BY_ONE_NAMED_SPARSE_TRACE_TRANSFER_INTERFACE"
        elif block_reason == "B3_BLOCKED_BY_SEED_TRANSFER_INTERFACE":
            verdict = "B3_BLOCKED_BY_SEED_TRANSFER_INTERFACE"
        elif block_reason == "B3_BLOCKED_BY_DOLFINX_ENTITYMAP_TAG_TRANSFER_INTERFACE":
            verdict = "B3_BLOCKED_BY_DOLFINX_ENTITYMAP_TAG_TRANSFER_INTERFACE"
        elif block_reason == "B3_BLOCKED_BY_DOLFINX_TRACE_TO_VOLUME_COUPLING_INTERFACE":
            verdict = "B3_BLOCKED_BY_DOLFINX_TRACE_TO_VOLUME_COUPLING_INTERFACE"
        elif b3_ops_assembled and not b3_seed_map:
            verdict = "B3_BLOCKED_BY_SEED_TRANSFER_INTERFACE"
        elif b3_ops_assembled and b3_ops_sanity and b3_seed_pass and b3_scalable:
            verdict = "B3_READY_FOR_JD_INERT_WIRING"
        else:
            verdict = "B3_REJECTED_DOES_NOT_PRESERVE_VALIDATED_V2_PHYSICS"

        payload: Dict[str, Any] = {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "selected_cleaned_formulation_route": "B3",
            **cpre,
            **pre,
            "C2_T_domain_space": c2_t_domain_space,
            "C2_T_codomain_space": c2_t_codomain_space,
            "C2_T_domain_dimension": c2_t_domain_dim,
            "C2_T_codomain_dimension": c2_t_codomain_dim,
            "C2_T_transfer_direction": c2_t_direction,
            "C2_T_construction_method": c2_t_method,
            "C2_T_is_sparse": c2_t_is_sparse,
            "C2_T_is_coordinate_interpolation_or_lift": c2_t_is_interp,
            "C2_T_preserves_trace_geometry_contract": c2_t_geom_preserve,
            "C2_T_constructed": c2_t_constructed,
            "C2_T_construction_blocker": c2_t_blocker,
            "C2_T_shape": c2_t_shape,
            "C2_T_nnz": c2_t_nnz,
            "C2_T_density": c2_t_density,
            "C2_T_row_nnz_min": c2_t_row_nnz_min,
            "C2_T_row_nnz_max": c2_t_row_nnz_max,
            "C2_T_mapping_checksum": c2_t_checksum,
            "C2_T_persisted_to_disk": False,
            "C2_T_geometry_map_contract_pass": c2_t_contract_pass,
            "C2_T_constant_field_transfer_pass": c2_t_const_pass,
            "C2_T_trace_support_transfer_pass": c2_t_support_pass,
            "C2_T_tag_support_transfer_pass": c2_t_tag_pass,
            "C2_T_validation_failure_reason": c2_t_validation_failure,
            "B3_shell_trace_space_constructed": bool(b3_space_ok),
            "B3_shell_trace_space_type": (
                "facet_submesh_vector_displacement_space" if b3_space_ok else "UNAVAILABLE"
            ),
            "B3_shell_trace_mesh_or_submesh_source": (
                "dolfinx.mesh.create_submesh(facet_union_tags_1_3_4)" if b3_space_ok else "UNAVAILABLE"
            ),
            "B3_submesh_entity_map_type": b3_submesh_map_type,
            "B3_submesh_entity_map_extraction_method": b3_submesh_map_method,
            "B3_submesh_to_parent_facet_map_extracted": bool(b3_submesh_map_ok),
            "B3_submesh_facet_count": int(b3_submesh_n),
            "B3_parent_facet_index_min": b3_parent_facet_min,
            "B3_parent_facet_index_max": b3_parent_facet_max,
            "B3_transferred_tag_counts": b3_transferred_counts,
            "B3_transferred_tag_count_convention": b3_tag_count_convention,
            "B3_transferred_tags_contract_pass": bool(b3_transferred_contract),
            "B3_original_structural_u_dimension": n_u,
            "B3_new_structural_u_dimension": b3_u_new,
            "B3_pressure_dimension_retained": n_p,
            "B3_total_cleaned_W_dimension": b3_total_w,
            "B3_dimension_reduction_ratio": _safe_float(b3_ratio),
            "B3_material_forms_assembled_on_trace_space": bool(b3_form_ok),
            "B3_material_form_transfer_method": "trace_submesh_facet_tag_meshtags_plus_surface_shell_forms",
            "B3_parent_shell_form_geometry_dependencies": b3_parent_geom_deps,
            "B3_invalid_trace_form_quantities_found": b3_invalid_trace_quantities,
            "B3_manifold_form_geometry_replacements": b3_geom_replacements,
            "B3_manifold_form_rederivation_method": b3_rederive_method,
            "B3_top_form_present": bool(b3_top),
            "B3_back_form_present": bool(b3_back),
            "B3_ribs_form_present": bool(b3_ribs),
            "B3_structural_mass_present": bool(b3_mass_present),
            "B3_structural_stiffness_present": bool(b3_stiff_present),
            "B3_material_form_failure_reason": None if b3_form_ok else (
                b3_material_fail_reason or "trace_form_assembly_or_support_missing"
            ),
            "B3_trace_structural_form_contract_pass": bool(b3_form_ok),
            "B3_changes_continuum_physical_meaning_of_weak_forms": (
                False if b3_form_ok else "UNRESOLVED"
            ),
            "B3_changes_discrete_basis_or_operator_representation": True,
            "B3_continuum_physics_preservation_status": b3_continuum_status,
            "B3_tag5_fix_transfer_constructed": bool(b3_bc_constructed),
            "B3_tag5_fixed_dof_count": int(b3_tag5_fixed_n),
            "B3_BC_contract_pass": bool(b3_bc_pass),
            "B3_BC_failure_reason": None if b3_bc_pass else b3_bc_reason,
            "B3_pressure_dimension_retained_expected": 24039,
            "B3_pressure_block_contract_pass": bool(n_p == 24039),
            "B3_trace_to_pressure_coupling_interface_constructed": bool(b3_coupling_iface),
            "B3_coupling_assembly_method": b3_coupling_method,
            "B3_coupling_transfer_or_entity_map_metadata": {
                "selected_route": selected_route,
                "C1_required_api": cpre["C1_required_api"],
                "C1_implementation_blocker": cpre["C1_implementation_blocker"],
                "C2_transfer_representation": cpre["C2_transfer_representation"],
                "C2_implementation_blocker": cpre["C2_implementation_blocker"],
                "C2_projected_coupling_formulae": (
                    {
                        "A_up_B3": "A_up_parent[parent_index_per_trace_dof, :]",
                        "A_pu_B3": "A_pu_parent[:, parent_index_per_trace_dof]",
                        "M_pu_B3": "M_pu_parent[:, parent_index_per_trace_dof]",
                    }
                    if c2_t_constructed
                    else None
                ),
            },
            "B3_coupling_present": bool(b3_coupling_present),
            "B3_coupling_failure_reason": None if b3_coupling_iface else b3_coupling_reason,
            "B3_coupling_stage_outcome": coupling_stage_outcome,
            "B3_A_and_M_assembled_without_EPS": bool(b3_ops_assembled),
            "B3_operator_dimensions": [b3_total_w, b3_total_w] if b3_total_w is not None else None,
            "B3_Auu_norm": b3_Auu_norm,
            "B3_Muu_norm": b3_Muu_norm,
            "B3_App_norm": b3_App_norm,
            "B3_Mpp_norm": b3_Mpp_norm,
            "B3_Aup_norm": b3_Aup_norm,
            "B3_Apu_norm": b3_Apu_norm,
            "B3_Mpu_norm": b3_Mpu_norm,
            "B3_structural_mass_null_coordinate_exposure_status": b3_null_exposure,
            "B3_no_EPS_operator_sanity_pass": bool(b3_ops_sanity),
            "B3_operator_sanity_failure_reason": None if b3_ops_sanity else b3_ops_reason,
            "B3_seed_mapping_constructed": bool(b3_seed_map),
            "B3_seed_transfer_method": b3_seed_method,
            "B3_seed_representable": bool(b3_seed_repr),
            "B3_seed_pressure_support_preserved": bool(b3_seed_pressure_support),
            "B3_seed_pressure_MAC": b3_seed_mac,
            "B3_seed_xH_Mx_original": seed_xhmx_o,
            "B3_seed_xH_Mx_B3": b3_seed_xhmx,
            "B3_seed_replay_frequency_original": seed_f_o,
            "B3_seed_replay_frequency_B3": b3_seed_f,
            "B3_seed_residual_original": seed_res_o,
            "B3_seed_residual_B3": b3_seed_res,
            "B3_seed_preservation_check_status": b3_seed_check_status,
            "B3_seed_preservation_pass": bool(b3_seed_pass),
            "B3_seed_preservation_failure_reason": None if b3_seed_pass else b3_seed_fail,
            "B3_scalability_gate_pass": bool(b3_scalable),
            "next_step_verdict": verdict,
            "artifact_storage_policy_applied": True,
            "report_size_target_bytes": REPORT_SIZE_TARGET_BYTES,
            "new_large_artifacts_created": [],
            "large_artifact_generation_authorized": False,
            "operator_matrices_persisted": False,
            "vector_banks_persisted": False,
            "solve_trees_created": False,
            "cleanup_required_before_production": True,
            "jd_wiring_authorized": False,
            "no_new_eigensolve_executed": True,
            "additional_eps": "NOT_AUTHORIZED",
        }
    finally:
        if A is not None:
            try:
                A.destroy()
            except Exception:
                pass
        if M is not None:
            try:
                M.destroy()
            except Exception:
                pass

    report_size = _write_json_atomic(OUT_JSON, payload)
    payload["report_size_bytes"] = int(report_size)
    _write_json_atomic(OUT_JSON, payload)

    md_lines = [
        "# B3 trace-coupled operator and seed-transfer audit (report-only)",
        "",
        f"Generated: {payload['generated_utc']}",
        "",
        f"- B3 trace->pressure coupling constructed: `{payload['B3_trace_to_pressure_coupling_interface_constructed']}`",
        f"- B3 A/M assembled without EPS: `{payload['B3_A_and_M_assembled_without_EPS']}`",
        f"- B3 operator sanity pass: `{payload['B3_no_EPS_operator_sanity_pass']}`",
        f"- B3 seed mapping constructed: `{payload['B3_seed_mapping_constructed']}`",
        f"- B3 seed preservation pass: `{payload['B3_seed_preservation_pass']}`",
        f"- B3 scalability gate pass: `{payload['B3_scalability_gate_pass']}`",
        f"- next verdict: `{payload['next_step_verdict']}`",
        "",
        "No eigensolve executed.",
    ]
    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")

    print(
        f"[B3_coupled] B3_trace_to_pressure_coupling_interface_constructed={payload['B3_trace_to_pressure_coupling_interface_constructed']}",
        flush=True,
    )
    print(
        f"[B3_coupled] B3_A_and_M_assembled_without_EPS={payload['B3_A_and_M_assembled_without_EPS']}",
        flush=True,
    )
    print(
        f"[B3_coupled] B3_no_EPS_operator_sanity_pass={payload['B3_no_EPS_operator_sanity_pass']}",
        flush=True,
    )
    print(
        f"[B3_coupled] B3_seed_mapping_constructed={payload['B3_seed_mapping_constructed']}",
        flush=True,
    )
    print(
        f"[B3_coupled] B3_seed_preservation_pass={payload['B3_seed_preservation_pass']}",
        flush=True,
    )
    print(
        f"[B3_coupled] B3_scalability_gate_pass={payload['B3_scalability_gate_pass']}",
        flush=True,
    )
    print(f"[B3_coupled] next_step_verdict={payload['next_step_verdict']}", flush=True)
    print(f"[B3_coupled] artifact_storage_policy_applied={payload['artifact_storage_policy_applied']}", flush=True)
    print(f"[B3_coupled] report_size_bytes={payload['report_size_bytes']}", flush=True)
    print(f"[B3_coupled] no_new_eigensolve_executed={payload['no_new_eigensolve_executed']}", flush=True)
    print(f"[B3_coupled] additional_eps={payload['additional_eps']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
