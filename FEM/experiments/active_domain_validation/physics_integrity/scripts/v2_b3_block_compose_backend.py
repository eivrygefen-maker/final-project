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
    del child_refs

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
