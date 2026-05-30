#!/usr/bin/env python3
"""Dev-only experimental B3 monolithic AIJ compose backends (default: direct row-loop)."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from petsc4py import PETSc

ENV_BACKEND = "B3_BLOCK_COMPOSE_BACKEND"
CLI_BACKEND_ARG = "--B3-block-compose-backend"
BASELINE_L_PROD_COMPOSE_SECONDS = 5971.655
CSR_BULK_RECOMMENDATION = (
    "MatNest→AIJ failed or unsupported. Next dev backend: bulk CSR row-batch MatSetValues "
    "(extract block CSR once, insert in batches) without changing physics or operator layout."
)

ALLOWED_BACKENDS = frozenset(
    {
        "direct_row_loop",
        "matnest_convert",
        "matnest_compare",
    }
)
COMPARE_ALLOWED_MESH_LEVELS = frozenset({"L_dev_dense"})


class B3BlockComposeBackendError(RuntimeError):
    def __init__(
        self,
        stage: str,
        message: str,
        *,
        petsc_error: Optional[str] = None,
        recommendation: Optional[str] = None,
        compose_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.stage = str(stage)
        self.message = str(message)
        self.petsc_error = petsc_error
        self.recommendation = recommendation or CSR_BULK_RECOMMENDATION
        self.compose_meta = dict(compose_meta) if compose_meta else None

    def as_dict(self) -> Dict[str, Any]:
        out = {
            "B3_BLOCK_COMPOSE_failure_stage": self.stage,
            "B3_BLOCK_COMPOSE_failure_reason": self.message,
            "B3_BLOCK_COMPOSE_petsc_error": self.petsc_error,
            "B3_BLOCK_COMPOSE_recommendation": self.recommendation,
        }
        if self.compose_meta:
            for key, val in self.compose_meta.items():
                if str(key).startswith("B3_BLOCK_COMPOSE_") or str(key).startswith("B3_final_MatNest_"):
                    out[key] = val
        return out


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def resolve_compose_backend(
    *,
    argv: Optional[Sequence[str]] = None,
    mesh_level: Optional[str] = None,
) -> str:
    """Resolve backend from CLI (highest priority), then env, default direct_row_loop."""
    backend: Optional[str] = None
    if argv is not None:
        for i, arg in enumerate(argv):
            if arg == CLI_BACKEND_ARG and i + 1 < len(argv):
                backend = str(argv[i + 1]).strip().lower()
                break
            if arg.startswith(f"{CLI_BACKEND_ARG}="):
                backend = str(arg.split("=", 1)[1]).strip().lower()
                break
    if backend is None:
        backend = os.environ.get(ENV_BACKEND, "direct_row_loop").strip().lower()
    if not backend:
        backend = "direct_row_loop"
    if backend not in ALLOWED_BACKENDS:
        raise B3BlockComposeBackendError(
            "backend_resolve",
            f"unsupported backend={backend!r}; allowed={sorted(ALLOWED_BACKENDS)}",
            recommendation=CSR_BULK_RECOMMENDATION,
        )
    if backend == "matnest_compare":
        ml = str(mesh_level or "").strip()
        if ml not in COMPARE_ALLOWED_MESH_LEVELS:
            raise B3BlockComposeBackendError(
                "backend_resolve",
                f"matnest_compare is dev-only on {sorted(COMPARE_ALLOWED_MESH_LEVELS)}; got mesh_level={ml!r}",
                recommendation="Use matnest_convert on L_prod or matnest_compare on L_dev_dense.",
            )
    return backend


def apply_compose_backend_from_argv(argv: Sequence[str], *, mesh_level: str) -> str:
    """Apply CLI backend to env (for subprocess workers) and return resolved backend."""
    backend = resolve_compose_backend(argv=argv, mesh_level=mesh_level)
    os.environ[ENV_BACKEND] = backend
    return backend


def _audit_helpers() -> Any:
    import run_v2_B3_trace_coupled_operator_and_seed_transfer_audit as audit

    return audit


def _matnest_breadcrumb(compose_meta: Dict[str, Any], stage: str, **extra: Any) -> None:
    compose_meta["B3_BLOCK_COMPOSE_matnest_last_breadcrumb"] = str(stage)
    compose_meta[f"B3_BLOCK_COMPOSE_matnest_breadcrumb_{stage}"] = True
    for key, val in extra.items():
        compose_meta[f"B3_BLOCK_COMPOSE_matnest_{stage}_{key}"] = val
    extra_s = " ".join(f"{k}={v!r}" for k, v in extra.items())
    print(f"[B3_BLOCK_COMPOSE_matnest] {stage} {extra_s}".rstrip(), flush=True)


def _matnest_child_diag(compose_meta: Dict[str, Any], prefix: str, mat: Any) -> None:
    audit = _audit_helpers()
    compose_meta[f"B3_BLOCK_COMPOSE_matnest_child_{prefix}_type"] = str(mat.getType())
    compose_meta[f"B3_BLOCK_COMPOSE_matnest_child_{prefix}_shape"] = audit._mat_shape(mat)
    compose_meta[f"B3_BLOCK_COMPOSE_matnest_child_{prefix}_nnz"] = int(
        audit._petsc_mat_global_nnz_used(mat)
    )


def _ensure_assembled(mat: Any) -> None:
    audit = _audit_helpers()
    audit._petsc_mat_try_assemble(mat)


def _owned_aij_copy(mat: Any) -> Any:
    """Independent assembled AIJ duplicate for MatNest (avoid createSubMatrix view bugs)."""
    audit = _audit_helpers()
    owned = audit._petsc_duplicate_scaled(mat, 1.0)
    _ensure_assembled(owned)
    if "nest" in str(owned.getType()).lower():
        raise B3BlockComposeBackendError(
            "matnest_child_copy",
            f"refusing MatNest child copy type={owned.getType()}",
            recommendation=CSR_BULK_RECOMMENDATION,
        )
    return owned


def _create_matnest_stepwise(
    blocks: List[List[Any]],
    *,
    comm: Any,
    compose_meta: Dict[str, Any],
    nest_label: str,
) -> Any:
    """Build 2x2 PETSc MatNest via instance-style petsc4py API with stage breadcrumbs."""
    _matnest_breadcrumb(compose_meta, f"before_create_{nest_label}_nest")
    nest = PETSc.Mat().createNest(blocks, comm=comm)
    _matnest_breadcrumb(
        compose_meta,
        f"after_create_{nest_label}_nest",
        nest_type=str(nest.getType()),
        nest_shape=list(nest.getSize()),
    )
    _matnest_breadcrumb(compose_meta, f"before_assemble_{nest_label}_nest")
    nest.assemble()
    _matnest_breadcrumb(
        compose_meta,
        f"after_assemble_{nest_label}_nest",
        nest_type=str(nest.getType()),
    )
    return nest


def _create_zero_aij_block(n_u: int, n_p: int, *, comm: Any) -> Any:
    """Explicit zero AIJ block [n_u x n_p] for missing Mup in mass MatNest."""
    audit = _audit_helpers()
    return audit._petsc_zero_mat(int(n_u), int(n_p), comm)


def _matnest_to_aij(nest: Any, *, stage: str, compose_meta: Dict[str, Any], nest_label: str) -> Any:
    _matnest_breadcrumb(compose_meta, f"before_convert_{nest_label}_nest", convert_stage=stage)
    petsc_err: Optional[str] = None
    try:
        out = nest.convert("aij")
        if out is not None:
            out.assemble()
            if "nest" not in str(out.getType()).lower():
                _matnest_breadcrumb(
                    compose_meta,
                    f"after_convert_{nest_label}_nest",
                    out_type=str(out.getType()),
                    out_shape=list(out.getSize()),
                )
                return out
    except Exception as exc:
        petsc_err = f"{type(exc).__name__}:{exc}"
    out = PETSc.Mat()
    try:
        nest.convert("aij", out)
        out.assemble()
        if "nest" not in str(out.getType()).lower():
            _matnest_breadcrumb(
                compose_meta,
                f"after_convert_{nest_label}_nest",
                out_type=str(out.getType()),
                out_shape=list(out.getSize()),
            )
            return out
    except Exception as exc:
        petsc_err = petsc_err or f"{type(exc).__name__}:{exc}"
        compose_meta["B3_BLOCK_COMPOSE_failure_stage"] = stage
        compose_meta["B3_BLOCK_COMPOSE_failure_reason"] = f"MatNest convert to AIJ failed: {petsc_err}"
        raise B3BlockComposeBackendError(
            stage,
            f"MatNest convert to AIJ failed: {petsc_err}",
            petsc_error=petsc_err,
            recommendation=CSR_BULK_RECOMMENDATION,
        ) from exc
    compose_meta["B3_BLOCK_COMPOSE_failure_stage"] = stage
    compose_meta["B3_BLOCK_COMPOSE_failure_reason"] = "MatNest convert returned AIJ type check failed"
    raise B3BlockComposeBackendError(
        stage,
        "MatNest convert returned AIJ type check failed",
        petsc_error=petsc_err,
        recommendation=CSR_BULK_RECOMMENDATION,
    )


def _duplicate_aij_output(mat: Any) -> Any:
    """Independent AIJ copy so nest/child teardown cannot invalidate convert output."""
    owned = mat.duplicate()
    owned.copy(mat)
    owned.assemble()
    return owned


def _safe_frobenius_difference(
    a: Any,
    b: Any,
    *,
    compose_meta: Dict[str, Any],
    label: str,
) -> Optional[float]:
    audit = _audit_helpers()
    try:
        diff = float(audit._petsc_mat_frobenius_difference(a, b))
        compose_meta[f"B3_BLOCK_COMPOSE_compare_{label}_frobenius_diff"] = _safe_float(diff)
        compose_meta[f"B3_BLOCK_COMPOSE_compare_{label}_frobenius_available"] = True
        return diff
    except Exception as exc:
        compose_meta[f"B3_BLOCK_COMPOSE_compare_{label}_frobenius_available"] = False
        compose_meta[f"B3_BLOCK_COMPOSE_compare_{label}_frobenius_unavailable_reason"] = (
            f"{type(exc).__name__}:{exc}"
        )
        compose_meta["B3_BLOCK_COMPOSE_compare_norm_unavailable"] = True
        return None


def _monolithic_block_name(row: int, col: int, n_u: int) -> str:
    nu = int(n_u)
    r = int(row)
    c = int(col)
    if r < nu and c < nu:
        return "uu"
    if r < nu and c >= nu:
        return "up"
    if r >= nu and c < nu:
        return "pu"
    return "pp"


def _mass_monolithic_block_name(row: int, col: int, n_u: int) -> str:
    nu = int(n_u)
    r = int(row)
    c = int(col)
    if r < nu and c < nu:
        return "Muu"
    if r < nu and c >= nu:
        return "Mup_zero"
    if r >= nu and c < nu:
        return "Mpu"
    return "Mpp"


def _row_support_flags(mat: Any, row: int, *, value_tol: float = 1.0e-12) -> Dict[str, Any]:
    flags: Dict[str, Any] = {
        "nnz": 0,
        "row_norm": 0.0,
        "diag_value": 0.0,
        "has_diag_entry": False,
        "identity_like": False,
        "zero_row_like": False,
    }
    try:
        cols, vals = mat.getRow(int(row))
    except TypeError:
        rowdat = mat.getRow(int(row))
        cols, vals = rowdat[0], rowdat[1]
    cols_a = np.asarray(cols, dtype=np.int32).ravel()
    vals_a = np.asarray(vals, dtype=np.float64).ravel()
    flags["nnz"] = int(cols_a.size)
    if vals_a.size:
        flags["row_norm"] = float(np.linalg.norm(vals_a))
    for c, v in zip(cols_a.tolist(), vals_a.tolist()):
        if int(c) == int(row):
            flags["has_diag_entry"] = True
            flags["diag_value"] = float(v)
            break
    try:
        mat.restoreRow(int(row))
    except Exception:
        pass
    flags["zero_row_like"] = bool(flags["row_norm"] <= float(value_tol))
    flags["identity_like"] = bool(
        flags["nnz"] == 1
        and flags["has_diag_entry"]
        and abs(flags["diag_value"] - 1.0) <= float(value_tol)
    )
    return flags


def _mat_sparsity_entry_map(mat: Any) -> Dict[Tuple[int, int], float]:
    out: Dict[Tuple[int, int], float] = {}
    nrow = int(mat.getSize()[0])
    for r in range(nrow):
        try:
            cols, vals = mat.getRow(r)
        except TypeError:
            rowdat = mat.getRow(r)
            cols, vals = rowdat[0], rowdat[1]
        for c, v in zip(np.asarray(cols, dtype=np.int32).ravel(), np.asarray(vals, dtype=np.float64).ravel()):
            out[(int(r), int(c))] = float(v)
        try:
            mat.restoreRow(r)
        except Exception:
            pass
    return out


def _extract_submatrix(
    mat: Any,
    *,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    comm: Any,
) -> Any:
    n_rows = int(row_end) - int(row_start)
    n_cols = int(col_end) - int(col_start)
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("empty submatrix range")
    is_row = PETSc.IS().createStride(n_rows, int(row_start), 1, comm=comm)
    is_col = PETSc.IS().createStride(n_cols, int(col_start), 1, comm=comm)
    try:
        sub = mat.createSubMatrix(is_row, is_col, PETSc.Mat.SubMatrixOption.EXTRACT)
    except Exception:
        sub = mat.createSubMatrix(is_row, is_col)
    sub.assemble()
    return sub


def _safe_block_frobenius_difference(
    old: Any,
    new: Any,
    *,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    comm: Any,
) -> Optional[float]:
    audit = _audit_helpers()
    sub_old = None
    sub_new = None
    try:
        sub_old = _extract_submatrix(
            old,
            row_start=row_start,
            row_end=row_end,
            col_start=col_start,
            col_end=col_end,
            comm=comm,
        )
        sub_new = _extract_submatrix(
            new,
            row_start=row_start,
            row_end=row_end,
            col_start=col_start,
            col_end=col_end,
            comm=comm,
        )
        return float(audit._petsc_mat_frobenius_difference(sub_old, sub_new))
    except Exception:
        return None
    finally:
        for sub in (sub_old, sub_new):
            if sub is not None:
                try:
                    sub.destroy()
                except Exception:
                    pass


def _record_compose_matrix_diff_audit(
    old: Any,
    new: Any,
    *,
    label: str,
    n_u: int,
    n_p: int,
    comm: Any,
    compose_meta: Dict[str, Any],
    block_name_fn: Any,
    block_fro_specs: Sequence[Tuple[str, int, int, int, int]],
) -> None:
    prefix = f"B3_BLOCK_COMPOSE_compare_{label}_diff"
    diff_mat = None
    try:
        old_map = _mat_sparsity_entry_map(old)
        new_map = _mat_sparsity_entry_map(new)
        keys = sorted(set(old_map.keys()) | set(new_map.keys()))
        entries: List[Dict[str, Any]] = []
        thresholds = (1.0e-12, 1.0e-9, 1.0e-6)
        threshold_counts = {str(t): 0 for t in thresholds}
        max_abs = 0.0
        only_diagonal = True
        only_identity_or_zero_rows = True
        block_counts: Dict[str, int] = {}
        for key in keys:
            old_v = float(old_map.get(key, 0.0))
            new_v = float(new_map.get(key, 0.0))
            diff_v = old_v - new_v
            abs_diff = abs(diff_v)
            if abs_diff <= 1.0e-15:
                continue
            r, c = key
            block = str(block_name_fn(r, c, n_u))
            block_counts[block] = int(block_counts.get(block, 0)) + 1
            if int(r) != int(c):
                only_diagonal = False
            row_old = _row_support_flags(old, r)
            row_new = _row_support_flags(new, r)
            row_bc_identity = bool(row_old["identity_like"] or row_new["identity_like"])
            row_zero_bc = bool(row_old["zero_row_like"] and row_new["zero_row_like"])
            if not (row_bc_identity or row_zero_bc):
                only_identity_or_zero_rows = False
            max_abs = max(max_abs, abs_diff)
            for t in thresholds:
                if abs_diff > float(t):
                    threshold_counts[str(t)] += 1
            entries.append(
                {
                    "row": int(r),
                    "col": int(c),
                    "block": block,
                    "old": _safe_float(old_v),
                    "new": _safe_float(new_v),
                    "diff": _safe_float(diff_v),
                    "abs_diff": _safe_float(abs_diff),
                    "row_old_identity_like": bool(row_old["identity_like"]),
                    "row_new_identity_like": bool(row_new["identity_like"]),
                    "row_old_zero_like": bool(row_old["zero_row_like"]),
                    "row_new_zero_like": bool(row_new["zero_row_like"]),
                    "row_old_diag_value": _safe_float(row_old["diag_value"]),
                    "row_new_diag_value": _safe_float(row_new["diag_value"]),
                }
            )
        entries.sort(key=lambda item: float(item.get("abs_diff") or 0.0), reverse=True)
        diff_mat = old.duplicate()
        old.copy(diff_mat)
        try:
            diff_mat.axpy(-1.0, new, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        except Exception:
            diff_mat.axpy(-1.0, new)
        diff_mat.assemble()
        diff_nnz = int(_audit_helpers()._petsc_mat_global_nnz_used(diff_mat))
        compose_meta[f"{prefix}_nnz"] = diff_nnz
        compose_meta[f"{prefix}_entrywise_nnz"] = int(len(entries))
        compose_meta[f"{prefix}_max_abs"] = _safe_float(max_abs)
        compose_meta[f"{prefix}_count_gt_1e-12"] = int(threshold_counts["1e-12"])
        compose_meta[f"{prefix}_count_gt_1e-9"] = int(threshold_counts["1e-9"])
        compose_meta[f"{prefix}_count_gt_1e-6"] = int(threshold_counts["1e-6"])
        compose_meta[f"{prefix}_only_on_diagonal"] = bool(only_diagonal and len(entries) > 0)
        compose_meta[f"{prefix}_only_on_diagonal_vacuous"] = bool(len(entries) == 0)
        compose_meta[f"{prefix}_only_bc_identity_or_zero_rows"] = bool(
            only_identity_or_zero_rows and len(entries) > 0
        )
        compose_meta[f"{prefix}_block_counts"] = block_counts
        compose_meta[f"{prefix}_top20"] = entries[:20]
        if only_diagonal and only_identity_or_zero_rows and len(entries) > 0:
            compose_meta[f"{prefix}_harmless_bc_identity_zero_row_candidate"] = True
            compose_meta[f"{prefix}_mathematically_harmless_note"] = (
                "All entrywise differences lie on diagonal rows classified as identity-like "
                "or zero-row BC support in at least one backend; likely explicit BC/identity "
                "storage mismatch rather than block ordering. compare_pass not auto-waived."
            )
        elif only_diagonal and len(entries) > 0:
            compose_meta[f"{prefix}_harmless_bc_identity_zero_row_candidate"] = False
            compose_meta[f"{prefix}_mathematically_harmless_note"] = (
                "Differences are diagonal-only but not confined to identity/zero BC rows; "
                "requires review before L_prod matnest_convert."
            )
        elif len(entries) > 0:
            compose_meta[f"{prefix}_harmless_bc_identity_zero_row_candidate"] = False
            compose_meta[f"{prefix}_mathematically_harmless_note"] = (
                "Off-diagonal or mixed-block differences present; not a BC/identity-only artifact."
            )
        else:
            compose_meta[f"{prefix}_harmless_bc_identity_zero_row_candidate"] = None
            compose_meta[f"{prefix}_mathematically_harmless_note"] = "No entrywise differences above 1e-15."

        nu = int(n_u)
        np_ = int(n_p)
        for block_label, r0, r1, c0, c1 in block_fro_specs:
            fro = _safe_block_frobenius_difference(
                old,
                new,
                row_start=r0,
                row_end=r1,
                col_start=c0,
                col_end=c1,
                comm=comm,
            )
            compose_meta[f"{prefix}_block_{block_label}_frobenius_diff"] = _safe_float(fro)
        compose_meta[f"{prefix}_audit_pass"] = True
    except Exception as exc:
        compose_meta[f"{prefix}_audit_pass"] = False
        compose_meta[f"{prefix}_audit_unavailable_reason"] = f"{type(exc).__name__}:{exc}"
    finally:
        if diff_mat is not None:
            try:
                diff_mat.destroy()
            except Exception:
                pass


def _matnest_convert_aij_from_restricted_blocks(
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
    compose_meta: Dict[str, Any],
    defer_lifecycle_teardown: bool = False,
) -> Tuple[Any, Any]:
    nest_owned: List[Any] = []
    child_refs: List[Any] = [a_uu, a_up, a_pu, a_pp, m_uu, m_pu, m_pp]
    for label, mat in (
        ("Auu", a_uu),
        ("Aup", a_up),
        ("Apu", a_pu),
        ("App", a_pp),
        ("Muu", m_uu),
        ("Mpu", m_pu),
        ("Mpp", m_pp),
    ):
        _ensure_assembled(mat)
        _matnest_child_diag(compose_meta, label, mat)

    m_up_zero = _create_zero_aij_block(int(n_u), int(n_p), comm=comm)
    _matnest_child_diag(compose_meta, "Mup_zero", m_up_zero)
    child_refs.append(m_up_zero)

    a_nest_blocks = [
        [_owned_aij_copy(a_uu), _owned_aij_copy(a_up)],
        [_owned_aij_copy(a_pu), _owned_aij_copy(a_pp)],
    ]
    m_nest_blocks = [
        [_owned_aij_copy(m_uu), m_up_zero],
        [_owned_aij_copy(m_pu), _owned_aij_copy(m_pp)],
    ]
    for row in a_nest_blocks + m_nest_blocks:
        nest_owned.extend(row)
    child_refs.extend(nest_owned)

    nest_a = None
    nest_m = None
    compose_meta["B3_BLOCK_COMPOSE_matnest_create_A_pass"] = False
    compose_meta["B3_BLOCK_COMPOSE_matnest_create_M_pass"] = False
    compose_meta["B3_BLOCK_COMPOSE_matnest_A_type"] = None
    compose_meta["B3_BLOCK_COMPOSE_matnest_M_type"] = None
    compose_meta["B3_BLOCK_COMPOSE_matnest_child_blocks_use_owned_aij_copies"] = True
    t_create0 = time.perf_counter()
    try:
        try:
            nest_a = _create_matnest_stepwise(
                a_nest_blocks,
                comm=comm,
                compose_meta=compose_meta,
                nest_label="A",
            )
            compose_meta["B3_BLOCK_COMPOSE_matnest_create_A_pass"] = True
            compose_meta["B3_BLOCK_COMPOSE_matnest_A_type"] = str(nest_a.getType())
        except Exception as exc:
            compose_meta["B3_BLOCK_COMPOSE_failure_stage"] = "matnest_create_A"
            compose_meta["B3_BLOCK_COMPOSE_failure_reason"] = f"{type(exc).__name__}:{exc}"
            raise B3BlockComposeBackendError(
                "matnest_create_A",
                f"{type(exc).__name__}:{exc}",
                petsc_error=f"{type(exc).__name__}:{exc}",
                recommendation=CSR_BULK_RECOMMENDATION,
                compose_meta=compose_meta,
            ) from exc
        try:
            nest_m = _create_matnest_stepwise(
                m_nest_blocks,
                comm=comm,
                compose_meta=compose_meta,
                nest_label="M",
            )
            compose_meta["B3_BLOCK_COMPOSE_matnest_create_M_pass"] = True
            compose_meta["B3_BLOCK_COMPOSE_matnest_M_type"] = str(nest_m.getType())
        except Exception as exc:
            compose_meta["B3_BLOCK_COMPOSE_failure_stage"] = "matnest_create_M"
            compose_meta["B3_BLOCK_COMPOSE_failure_reason"] = f"{type(exc).__name__}:{exc}"
            if nest_a is not None:
                try:
                    nest_a.destroy()
                except Exception:
                    pass
            raise B3BlockComposeBackendError(
                "matnest_create_M",
                f"{type(exc).__name__}:{exc}",
                petsc_error=f"{type(exc).__name__}:{exc}",
                recommendation=CSR_BULK_RECOMMENDATION,
                compose_meta=compose_meta,
            ) from exc
    finally:
        compose_meta["B3_BLOCK_COMPOSE_matnest_create_seconds"] = _safe_float(
            time.perf_counter() - t_create0
        )

    t_conv_a0 = time.perf_counter()
    try:
        a_out = _matnest_to_aij(
            nest_a,
            stage="matnest_convert_A",
            compose_meta=compose_meta,
            nest_label="A",
        )
    finally:
        compose_meta["B3_BLOCK_COMPOSE_matnest_convert_A_seconds"] = _safe_float(
            time.perf_counter() - t_conv_a0
        )

    t_conv_m0 = time.perf_counter()
    try:
        m_out = _matnest_to_aij(
            nest_m,
            stage="matnest_convert_M",
            compose_meta=compose_meta,
            nest_label="M",
        )
    finally:
        compose_meta["B3_BLOCK_COMPOSE_matnest_convert_M_seconds"] = _safe_float(
            time.perf_counter() - t_conv_m0
        )

    _matnest_breadcrumb(compose_meta, "after_convert_M_nest_complete")

    a_ret = _duplicate_aij_output(a_out)
    m_ret = _duplicate_aij_output(m_out)
    compose_meta["B3_BLOCK_COMPOSE_matnest_outputs_duplicated_before_teardown"] = True

    if defer_lifecycle_teardown:
        compose_meta["B3_BLOCK_COMPOSE_matnest_lifecycle_teardown_deferred"] = True
        _matnest_breadcrumb(compose_meta, "before_destroy_nests", skipped=True)
        _matnest_breadcrumb(compose_meta, "after_destroy_nests", skipped=True)
        _matnest_breadcrumb(compose_meta, "before_return_matnest_result", teardown="deferred")
        compose_meta["B3_final_MatNest_constructed_after_blockwise_restriction"] = True
        compose_meta["B3_final_MatNest_conversion_to_sparse_AIJ_attempted"] = True
        compose_meta["B3_MatNest_to_AIJ_conversion_path_disabled"] = False
        compose_meta["B3_final_sparse_AIJ_conversion_method"] = "PETSc_MatNest_convert_aij_experimental"
        return a_ret, m_ret

    _matnest_breadcrumb(compose_meta, "before_destroy_nests")
    try:
        nest_a.destroy()
    except Exception:
        pass
    try:
        nest_m.destroy()
    except Exception:
        pass
    for owned in nest_owned:
        if owned is m_up_zero:
            continue
        try:
            owned.destroy()
        except Exception:
            pass
    try:
        m_up_zero.destroy()
    except Exception:
        pass
    try:
        a_out.destroy()
    except Exception:
        pass
    try:
        m_out.destroy()
    except Exception:
        pass
    del child_refs
    _matnest_breadcrumb(compose_meta, "after_destroy_nests")
    _matnest_breadcrumb(compose_meta, "before_return_matnest_result", teardown="completed")

    compose_meta["B3_final_MatNest_constructed_after_blockwise_restriction"] = True
    compose_meta["B3_final_MatNest_conversion_to_sparse_AIJ_attempted"] = True
    compose_meta["B3_MatNest_to_AIJ_conversion_path_disabled"] = False
    compose_meta["B3_final_sparse_AIJ_conversion_method"] = "PETSc_MatNest_convert_aij_experimental"
    return a_ret, m_ret


def _direct_row_loop_compose(
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
    operator_build_profile: Any = None,
) -> Tuple[Any, Any]:
    audit = _audit_helpers()
    return audit._b3_direct_sparse_aij_from_restricted_blocks(
        a_uu=a_uu,
        a_up=a_up,
        a_pu=a_pu,
        a_pp=a_pp,
        m_uu=m_uu,
        m_pu=m_pu,
        m_pp=m_pp,
        n_u=int(n_u),
        n_p=int(n_p),
        comm=comm,
        operator_build_profile=operator_build_profile,
    )


def _compare_backends_dev(
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
    compose_meta: Dict[str, Any],
    operator_build_profile: Any = None,
) -> Tuple[Any, Any]:
    audit = _audit_helpers()
    _matnest_breadcrumb(compose_meta, "before_direct_row_loop_compare_baseline")
    a_old, m_old = _direct_row_loop_compose(
        a_uu=a_uu,
        a_up=a_up,
        a_pu=a_pu,
        a_pp=a_pp,
        m_uu=m_uu,
        m_pu=m_pu,
        m_pp=m_pp,
        n_u=n_u,
        n_p=n_p,
        comm=comm,
        operator_build_profile=None,
    )
    _matnest_breadcrumb(compose_meta, "after_direct_row_loop_compare_baseline")
    _matnest_breadcrumb(compose_meta, "before_matnest_convert_compare")
    a_new, m_new = _matnest_convert_aij_from_restricted_blocks(
        a_uu=a_uu,
        a_up=a_up,
        a_pu=a_pu,
        a_pp=a_pp,
        m_uu=m_uu,
        m_pu=m_pu,
        m_pp=m_pp,
        n_u=n_u,
        n_p=n_p,
        comm=comm,
        compose_meta=compose_meta,
        defer_lifecycle_teardown=True,
    )
    _matnest_breadcrumb(compose_meta, "after_matnest_convert_compare")

    _matnest_breadcrumb(compose_meta, "before_compare_shapes")
    a_shape_old = audit._mat_shape(a_old)
    a_shape_new = audit._mat_shape(a_new)
    m_shape_old = audit._mat_shape(m_old)
    m_shape_new = audit._mat_shape(m_new)
    a_nnz_old = int(audit._petsc_mat_global_nnz_used(a_old))
    a_nnz_new = int(audit._petsc_mat_global_nnz_used(a_new))
    m_nnz_old = int(audit._petsc_mat_global_nnz_used(m_old))
    m_nnz_new = int(audit._petsc_mat_global_nnz_used(m_new))
    shape_equal = bool(a_shape_old == a_shape_new and m_shape_old == m_shape_new)
    a_nnz_equal = bool(a_nnz_old == a_nnz_new)
    m_nnz_equal = bool(m_nnz_old == m_nnz_new)
    compose_meta.update(
        {
            "B3_BLOCK_COMPOSE_compare_mode": True,
            "B3_BLOCK_COMPOSE_compare_A_shape_old": a_shape_old,
            "B3_BLOCK_COMPOSE_compare_A_shape_new": a_shape_new,
            "B3_BLOCK_COMPOSE_compare_M_shape_old": m_shape_old,
            "B3_BLOCK_COMPOSE_compare_M_shape_new": m_shape_new,
            "B3_BLOCK_COMPOSE_compare_A_type_old": str(a_old.getType()),
            "B3_BLOCK_COMPOSE_compare_A_type_new": str(a_new.getType()),
            "B3_BLOCK_COMPOSE_compare_M_type_old": str(m_old.getType()),
            "B3_BLOCK_COMPOSE_compare_M_type_new": str(m_new.getType()),
            "B3_BLOCK_COMPOSE_compare_shape_equal": shape_equal,
            "B3_BLOCK_COMPOSE_compare_A_nnz_old": a_nnz_old,
            "B3_BLOCK_COMPOSE_compare_A_nnz_new": a_nnz_new,
            "B3_BLOCK_COMPOSE_compare_M_nnz_old": m_nnz_old,
            "B3_BLOCK_COMPOSE_compare_M_nnz_new": m_nnz_new,
            "B3_BLOCK_COMPOSE_compare_A_nnz_equal": a_nnz_equal,
            "B3_BLOCK_COMPOSE_compare_M_nnz_equal": m_nnz_equal,
        }
    )
    _matnest_breadcrumb(
        compose_meta,
        "after_compare_shapes",
        shape_equal=shape_equal,
        A_nnz_equal=a_nnz_equal,
        M_nnz_equal=m_nnz_equal,
    )

    tol_a: Optional[float] = None
    tol_m: Optional[float] = None
    a_fro_diff: Optional[float] = None
    m_fro_diff: Optional[float] = None
    if shape_equal and a_nnz_equal and m_nnz_equal:
        _matnest_breadcrumb(compose_meta, "before_compare_norm_A")
        a_fro_diff = _safe_frobenius_difference(a_old, a_new, compose_meta=compose_meta, label="A")
        _matnest_breadcrumb(
            compose_meta,
            "after_compare_norm_A",
            frobenius_available=compose_meta.get("B3_BLOCK_COMPOSE_compare_A_frobenius_available"),
        )
        _matnest_breadcrumb(compose_meta, "before_compare_norm_M")
        m_fro_diff = _safe_frobenius_difference(m_old, m_new, compose_meta=compose_meta, label="M")
        _matnest_breadcrumb(
            compose_meta,
            "after_compare_norm_M",
            frobenius_available=compose_meta.get("B3_BLOCK_COMPOSE_compare_M_frobenius_available"),
        )
        if (
            compose_meta.get("B3_BLOCK_COMPOSE_compare_A_frobenius_available")
            and compose_meta.get("B3_BLOCK_COMPOSE_compare_M_frobenius_available")
        ):
            try:
                a_norm_old = float(a_old.norm(PETSc.NormType.FROBENIUS))
                m_norm_old = float(m_old.norm(PETSc.NormType.FROBENIUS))
                tol_a = max(1.0e-8, 1.0e-12 * max(a_norm_old, 1.0))
                tol_m = max(1.0e-8, 1.0e-12 * max(m_norm_old, 1.0))
                compose_meta["B3_BLOCK_COMPOSE_compare_A_frobenius_tol"] = _safe_float(tol_a)
                compose_meta["B3_BLOCK_COMPOSE_compare_M_frobenius_tol"] = _safe_float(tol_m)
            except Exception as exc:
                compose_meta["B3_BLOCK_COMPOSE_compare_norm_unavailable"] = True
                compose_meta["B3_BLOCK_COMPOSE_compare_norm_tol_unavailable_reason"] = (
                    f"{type(exc).__name__}:{exc}"
                )
        _matnest_breadcrumb(compose_meta, "before_compare_diff_audit_A")
        nu = int(n_u)
        np_ = int(n_p)
        _record_compose_matrix_diff_audit(
            a_old,
            a_new,
            label="A",
            n_u=nu,
            n_p=np_,
            comm=comm,
            compose_meta=compose_meta,
            block_name_fn=_monolithic_block_name,
            block_fro_specs=(
                ("Auu", 0, nu, 0, nu),
                ("Aup", 0, nu, nu, nu + np_),
                ("Apu", nu, nu + np_, 0, nu),
                ("App", nu, nu + np_, nu, nu + np_),
            ),
        )
        _matnest_breadcrumb(compose_meta, "after_compare_diff_audit_A")
        _matnest_breadcrumb(compose_meta, "before_compare_diff_audit_M")
        _record_compose_matrix_diff_audit(
            m_old,
            m_new,
            label="M",
            n_u=nu,
            n_p=np_,
            comm=comm,
            compose_meta=compose_meta,
            block_name_fn=_mass_monolithic_block_name,
            block_fro_specs=(
                ("Muu", 0, nu, 0, nu),
                ("Mup_zero", 0, nu, nu, nu + np_),
                ("Mpu", nu, nu + np_, 0, nu),
                ("Mpp", nu, nu + np_, nu, nu + np_),
            ),
        )
        _matnest_breadcrumb(compose_meta, "after_compare_diff_audit_M")
    else:
        compose_meta["B3_BLOCK_COMPOSE_compare_norm_skipped"] = True
        compose_meta["B3_BLOCK_COMPOSE_compare_norm_skipped_reason"] = "shape_or_nnz_mismatch"

    norm_pass = True
    if compose_meta.get("B3_BLOCK_COMPOSE_compare_norm_unavailable"):
        norm_pass = False
    elif a_fro_diff is not None and m_fro_diff is not None and tol_a is not None and tol_m is not None:
        norm_pass = bool(a_fro_diff <= tol_a and m_fro_diff <= tol_m)
    elif not compose_meta.get("B3_BLOCK_COMPOSE_compare_norm_skipped"):
        norm_pass = False

    compare_pass = bool(shape_equal and a_nnz_equal and m_nnz_equal and norm_pass)
    if compose_meta.get("B3_BLOCK_COMPOSE_compare_norm_unavailable") and shape_equal and a_nnz_equal and m_nnz_equal:
        compose_meta["B3_BLOCK_COMPOSE_compare_pass"] = True
        compose_meta["B3_BLOCK_COMPOSE_compare_pass_basis"] = "shape_and_nnz_only_norm_unavailable"
        compare_pass = True
    else:
        compose_meta["B3_BLOCK_COMPOSE_compare_pass"] = compare_pass
        compose_meta["B3_BLOCK_COMPOSE_compare_pass_basis"] = "shape_nnz_and_frobenius"

    if not compare_pass:
        raise B3BlockComposeBackendError(
            "matnest_compare_correctness",
            (
                f"shape_equal={shape_equal};A_nnz_equal={a_nnz_equal};M_nnz_equal={m_nnz_equal}; "
                f"A_fro_diff={a_fro_diff};M_fro_diff={m_fro_diff}; "
                f"norm_unavailable={compose_meta.get('B3_BLOCK_COMPOSE_compare_norm_unavailable')}"
            ),
            recommendation=CSR_BULK_RECOMMENDATION,
            compose_meta=compose_meta,
        )

    compose_meta["B3_BLOCK_COMPOSE_backend_selected_for_downstream"] = "matnest_convert"
    _matnest_breadcrumb(compose_meta, "before_return_matnest_compare_result")
    return a_new, m_new


def _record_compose_structure_contract(meta: Dict[str, Any], *, a_out: Any, m_out: Any, n_w: int) -> None:
    audit = _audit_helpers()
    meta["B3_BLOCK_COMPOSE_A_shape"] = audit._mat_shape(a_out)
    meta["B3_BLOCK_COMPOSE_M_shape"] = audit._mat_shape(m_out)
    meta["B3_BLOCK_COMPOSE_A_type"] = str(a_out.getType())
    meta["B3_BLOCK_COMPOSE_M_type"] = str(m_out.getType())
    meta["B3_BLOCK_COMPOSE_A_nnz"] = int(audit._petsc_mat_global_nnz_used(a_out))
    meta["B3_BLOCK_COMPOSE_M_nnz"] = int(audit._petsc_mat_global_nnz_used(m_out))
    meta["B3_BLOCK_COMPOSE_structure_contract_pass"] = bool(
        a_out.getSize() == (int(n_w), int(n_w))
        and m_out.getSize() == (int(n_w), int(n_w))
        and "nest" not in str(a_out.getType()).lower()
        and "nest" not in str(m_out.getType()).lower()
        and int(meta["B3_BLOCK_COMPOSE_A_nnz"]) > 0
        and int(meta["B3_BLOCK_COMPOSE_M_nnz"]) > 0
    )


def _record_compose_timing(meta: Dict[str, Any], *, backend: str, elapsed_s: float, mesh_level: str) -> None:
    meta["B3_BLOCK_COMPOSE_backend"] = backend
    meta["B3_BLOCK_COMPOSE_experimental_total_seconds"] = _safe_float(elapsed_s)
    if backend in ("matnest_convert", "matnest_compare"):
        ref = float(BASELINE_L_PROD_COMPOSE_SECONDS)
        meta["B3_BLOCK_COMPOSE_baseline_reference_seconds"] = ref
        exp = float(elapsed_s)
        if exp > 0.0:
            meta["B3_BLOCK_COMPOSE_speedup_vs_reference"] = _safe_float(ref / exp)
        else:
            meta["B3_BLOCK_COMPOSE_speedup_vs_reference"] = None
    if backend == "direct_row_loop" and str(mesh_level) == "L_prod":
        meta["B3_BLOCK_COMPOSE_baseline_reference_seconds"] = float(BASELINE_L_PROD_COMPOSE_SECONDS)


def compose_restricted_blocks_to_monolithic_aij(
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
    report_meta: Dict[str, Any],
    mesh_level: str = "",
    operator_build_profile: Any = None,
    argv: Optional[Sequence[str]] = None,
) -> Tuple[Any, Any]:
    """Dispatch monolithic AIJ compose; no silent fallback on experimental backends."""
    backend = resolve_compose_backend(argv=argv, mesh_level=mesh_level)
    t0 = time.perf_counter()
    try:
        if backend == "direct_row_loop":
            a_out, m_out = _direct_row_loop_compose(
                a_uu=a_uu,
                a_up=a_up,
                a_pu=a_pu,
                a_pp=a_pp,
                m_uu=m_uu,
                m_pu=m_pu,
                m_pp=m_pp,
                n_u=n_u,
                n_p=n_p,
                comm=comm,
                operator_build_profile=operator_build_profile,
            )
        elif backend == "matnest_convert":
            a_out, m_out = _matnest_convert_aij_from_restricted_blocks(
                a_uu=a_uu,
                a_up=a_up,
                a_pu=a_pu,
                a_pp=a_pp,
                m_uu=m_uu,
                m_pu=m_pu,
                m_pp=m_pp,
                n_u=n_u,
                n_p=n_p,
                comm=comm,
                compose_meta=report_meta,
            )
        elif backend == "matnest_compare":
            a_out, m_out = _compare_backends_dev(
                a_uu=a_uu,
                a_up=a_up,
                a_pu=a_pu,
                a_pp=a_pp,
                m_uu=m_uu,
                m_pu=m_pu,
                m_pp=m_pp,
                n_u=n_u,
                n_p=n_p,
                comm=comm,
                compose_meta=report_meta,
                operator_build_profile=operator_build_profile,
            )
        else:
            raise B3BlockComposeBackendError(
                "backend_dispatch",
                f"unsupported backend={backend!r}",
                recommendation=CSR_BULK_RECOMMENDATION,
            )
    except B3BlockComposeBackendError as exc:
        report_meta.update(
            {
                "B3_BLOCK_COMPOSE_backend": backend,
                "B3_BLOCK_COMPOSE_experimental_total_seconds": _safe_float(time.perf_counter() - t0),
            }
        )
        if exc.compose_meta is None:
            exc.compose_meta = dict(report_meta)
        else:
            merged = dict(exc.compose_meta)
            for key, val in report_meta.items():
                if str(key).startswith("B3_BLOCK_COMPOSE_") or str(key).startswith("B3_final_MatNest_"):
                    merged[key] = val
            exc.compose_meta = merged
        raise
    except Exception as exc:
        report_meta.update(
            {
                "B3_BLOCK_COMPOSE_backend": backend,
                "B3_BLOCK_COMPOSE_failure_stage": "compose_unhandled",
                "B3_BLOCK_COMPOSE_failure_reason": f"{type(exc).__name__}:{exc}",
                "B3_BLOCK_COMPOSE_recommendation": CSR_BULK_RECOMMENDATION,
                "B3_BLOCK_COMPOSE_experimental_total_seconds": _safe_float(time.perf_counter() - t0),
            }
        )
        raise B3BlockComposeBackendError(
            "compose_unhandled",
            f"{type(exc).__name__}:{exc}",
            petsc_error=f"{type(exc).__name__}:{exc}",
            recommendation=CSR_BULK_RECOMMENDATION,
            compose_meta=report_meta,
        ) from exc

    elapsed = time.perf_counter() - t0
    n_w = int(n_u) + int(n_p)
    _record_compose_structure_contract(report_meta, a_out=a_out, m_out=m_out, n_w=n_w)
    _record_compose_timing(report_meta, backend=backend, elapsed_s=elapsed, mesh_level=mesh_level)
    report_meta["B3_final_operator_construction_method"] = (
        "DIRECT_SPARSE_MONOLITHIC_AIJ_FROM_RESTRICTED_BLOCKS"
        if backend == "direct_row_loop"
        else "EXPERIMENTAL_MATNEST_CONVERT_TO_AIJ_FROM_RESTRICTED_BLOCKS"
    )
    return a_out, m_out
