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
    ) -> None:
        super().__init__(message)
        self.stage = str(stage)
        self.message = str(message)
        self.petsc_error = petsc_error
        self.recommendation = recommendation or CSR_BULK_RECOMMENDATION

    def as_dict(self) -> Dict[str, Any]:
        return {
            "B3_BLOCK_COMPOSE_failure_stage": self.stage,
            "B3_BLOCK_COMPOSE_failure_reason": self.message,
            "B3_BLOCK_COMPOSE_petsc_error": self.petsc_error,
            "B3_BLOCK_COMPOSE_recommendation": self.recommendation,
        }


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


def _create_nest_2x2(
    blocks: List[List[Any]],
    *,
    n_u: int,
    n_p: int,
    comm: Any,
) -> Any:
    is_u = PETSc.IS().createStride(int(n_u), 0, 1, comm=comm)
    is_p = PETSc.IS().createStride(int(n_p), 0, 1, comm=comm)
    try:
        nest = PETSc.Mat.createNest([is_u, is_p], [is_u, is_p], blocks, comm=comm)
        nest.assemble()
        return nest
    finally:
        is_u.destroy()
        is_p.destroy()


def _matnest_to_aij(nest: Any, *, stage: str) -> Any:
    petsc_err: Optional[str] = None
    try:
        out = nest.convert("aij")
        if out is not None:
            out.assemble()
            if "nest" not in str(out.getType()).lower():
                return out
    except Exception as exc:
        petsc_err = f"{type(exc).__name__}:{exc}"
    out = PETSc.Mat()
    try:
        nest.convert("aij", out)
        out.assemble()
        if "nest" not in str(out.getType()).lower():
            return out
    except Exception as exc:
        petsc_err = petsc_err or f"{type(exc).__name__}:{exc}"
        raise B3BlockComposeBackendError(
            stage,
            f"MatNest convert to AIJ failed: {petsc_err}",
            petsc_error=petsc_err,
            recommendation=CSR_BULK_RECOMMENDATION,
        ) from exc
    raise B3BlockComposeBackendError(
        stage,
        "MatNest convert returned AIJ type check failed",
        petsc_error=petsc_err,
        recommendation=CSR_BULK_RECOMMENDATION,
    )


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
) -> Tuple[Any, Any]:
    audit = _audit_helpers()
    m_up_zero = audit._petsc_zero_mat(int(n_u), int(n_p), comm)
    nest_a = None
    nest_m = None
    t_create0 = time.perf_counter()
    try:
        nest_a = _create_nest_2x2(
            [[a_uu, a_up], [a_pu, a_pp]],
            n_u=int(n_u),
            n_p=int(n_p),
            comm=comm,
        )
        nest_m = _create_nest_2x2(
            [[m_uu, m_up_zero], [m_pu, m_pp]],
            n_u=int(n_u),
            n_p=int(n_p),
            comm=comm,
        )
    except Exception as exc:
        raise B3BlockComposeBackendError(
            "matnest_create",
            f"{type(exc).__name__}:{exc}",
            petsc_error=f"{type(exc).__name__}:{exc}",
            recommendation=CSR_BULK_RECOMMENDATION,
        ) from exc
    finally:
        compose_meta["B3_BLOCK_COMPOSE_matnest_create_seconds"] = _safe_float(
            time.perf_counter() - t_create0
        )

    t_conv_a0 = time.perf_counter()
    try:
        a_out = _matnest_to_aij(nest_a, stage="matnest_convert_A")
    finally:
        compose_meta["B3_BLOCK_COMPOSE_matnest_convert_A_seconds"] = _safe_float(
            time.perf_counter() - t_conv_a0
        )

    t_conv_m0 = time.perf_counter()
    try:
        m_out = _matnest_to_aij(nest_m, stage="matnest_convert_M")
    finally:
        compose_meta["B3_BLOCK_COMPOSE_matnest_convert_M_seconds"] = _safe_float(
            time.perf_counter() - t_conv_m0
        )

    try:
        nest_a.destroy()
    except Exception:
        pass
    try:
        nest_m.destroy()
    except Exception:
        pass
    try:
        m_up_zero.destroy()
    except Exception:
        pass

    compose_meta["B3_final_MatNest_constructed_after_blockwise_restriction"] = True
    compose_meta["B3_final_MatNest_conversion_to_sparse_AIJ_attempted"] = True
    compose_meta["B3_MatNest_to_AIJ_conversion_path_disabled"] = False
    compose_meta["B3_final_sparse_AIJ_conversion_method"] = "PETSc_MatNest_convert_aij_experimental"
    return a_out, m_out


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
    )

    a_shape_old = audit._mat_shape(a_old)
    a_shape_new = audit._mat_shape(a_new)
    m_shape_old = audit._mat_shape(m_old)
    m_shape_new = audit._mat_shape(m_new)
    a_nnz_old = int(audit._petsc_mat_global_nnz_used(a_old))
    a_nnz_new = int(audit._petsc_mat_global_nnz_used(a_new))
    m_nnz_old = int(audit._petsc_mat_global_nnz_used(m_old))
    m_nnz_new = int(audit._petsc_mat_global_nnz_used(m_new))
    a_fro_diff = float(audit._petsc_mat_frobenius_difference(a_old, a_new))
    m_fro_diff = float(audit._petsc_mat_frobenius_difference(m_old, m_new))
    a_norm_old = float(a_old.norm(PETSc.NormType.FROBENIUS))
    m_norm_old = float(m_old.norm(PETSc.NormType.FROBENIUS))
    tol_a = max(1.0e-8, 1.0e-12 * max(a_norm_old, 1.0))
    tol_m = max(1.0e-8, 1.0e-12 * max(m_norm_old, 1.0))

    compose_meta.update(
        {
            "B3_BLOCK_COMPOSE_compare_mode": True,
            "B3_BLOCK_COMPOSE_compare_A_shape_old": a_shape_old,
            "B3_BLOCK_COMPOSE_compare_A_shape_new": a_shape_new,
            "B3_BLOCK_COMPOSE_compare_M_shape_old": m_shape_old,
            "B3_BLOCK_COMPOSE_compare_M_shape_new": m_shape_new,
            "B3_BLOCK_COMPOSE_compare_shape_equal": bool(
                a_shape_old == a_shape_new and m_shape_old == m_shape_new
            ),
            "B3_BLOCK_COMPOSE_compare_A_nnz_old": a_nnz_old,
            "B3_BLOCK_COMPOSE_compare_A_nnz_new": a_nnz_new,
            "B3_BLOCK_COMPOSE_compare_M_nnz_old": m_nnz_old,
            "B3_BLOCK_COMPOSE_compare_M_nnz_new": m_nnz_new,
            "B3_BLOCK_COMPOSE_compare_A_nnz_equal": bool(a_nnz_old == a_nnz_new),
            "B3_BLOCK_COMPOSE_compare_M_nnz_equal": bool(m_nnz_old == m_nnz_new),
            "B3_BLOCK_COMPOSE_compare_A_frobenius_diff": _safe_float(a_fro_diff),
            "B3_BLOCK_COMPOSE_compare_M_frobenius_diff": _safe_float(m_fro_diff),
            "B3_BLOCK_COMPOSE_compare_A_frobenius_tol": _safe_float(tol_a),
            "B3_BLOCK_COMPOSE_compare_M_frobenius_tol": _safe_float(tol_m),
        }
    )

    compare_pass = bool(
        compose_meta["B3_BLOCK_COMPOSE_compare_shape_equal"]
        and a_fro_diff <= tol_a
        and m_fro_diff <= tol_m
    )
    compose_meta["B3_BLOCK_COMPOSE_compare_pass"] = compare_pass
    if not compare_pass:
        try:
            a_old.destroy()
            m_old.destroy()
        except Exception:
            pass
        try:
            a_new.destroy()
            m_new.destroy()
        except Exception:
            pass
        raise B3BlockComposeBackendError(
            "matnest_compare_correctness",
            (
                f"shape_equal={compose_meta['B3_BLOCK_COMPOSE_compare_shape_equal']}; "
                f"A_nnz_old={a_nnz_old};A_nnz_new={a_nnz_new}; "
                f"M_nnz_old={m_nnz_old};M_nnz_new={m_nnz_new}; "
                f"A_fro_diff={a_fro_diff:.6e};M_fro_diff={m_fro_diff:.6e}"
            ),
            recommendation=CSR_BULK_RECOMMENDATION,
        )

    try:
        a_old.destroy()
        m_old.destroy()
    except Exception:
        pass
    compose_meta["B3_BLOCK_COMPOSE_backend_selected_for_downstream"] = "matnest_convert"
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
    except B3BlockComposeBackendError:
        report_meta.update(
            {
                "B3_BLOCK_COMPOSE_backend": backend,
                "B3_BLOCK_COMPOSE_experimental_total_seconds": _safe_float(time.perf_counter() - t0),
            }
        )
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
