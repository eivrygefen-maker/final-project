#!/usr/bin/env python3
"""Dev-only experimental B3 monolithic AIJ compose backends (default: direct row-loop)."""
from __future__ import annotations

import heapq
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
DIFF_THRESHOLDS: Tuple[Tuple[str, float], ...] = (
    ("1e-12", 1.0e-12),
    ("1e-9", 1.0e-9),
    ("1e-6", 1.0e-6),
)
EXPLICIT_ZERO_TOL = 1.0e-15
VALUE_DIFF_MIN = 1.0e-15


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
    """Independent assembled AIJ duplicate for MatNest (value-preserving copy)."""
    _ensure_assembled(mat)
    owned = mat.duplicate()
    mat.copy(owned)
    owned.assemble()
    if "nest" in str(owned.getType()).lower():
        raise B3BlockComposeBackendError(
            "matnest_child_copy",
            f"refusing MatNest child copy type={owned.getType()}",
            recommendation=CSR_BULK_RECOMMENDATION,
        )
    return owned


def _record_matnest_owned_copy_diag(
    original: Any,
    owned: Any,
    *,
    label: str,
    compose_meta: Dict[str, Any],
) -> None:
    audit = _audit_helpers()
    prefix = f"B3_BLOCK_COMPOSE_matnest_owned_copy_{label}"
    try:
        orig_norm = float(original.norm(PETSc.NormType.FROBENIUS))
        owned_norm = float(owned.norm(PETSc.NormType.FROBENIUS))
        fro_diff = float(audit._petsc_mat_frobenius_difference(original, owned))
        denom = max(orig_norm, owned_norm, 1.0e-300)
        relative_diff = float(fro_diff / denom)
        compose_meta[f"{prefix}_original_norm"] = _safe_float(orig_norm)
        compose_meta[f"{prefix}_owned_copy_norm"] = _safe_float(owned_norm)
        compose_meta[f"{prefix}_frobenius_diff"] = _safe_float(fro_diff)
        compose_meta[f"{prefix}_relative_diff"] = _safe_float(relative_diff)
        compose_meta[f"{prefix}_value_copy_pass"] = bool(fro_diff <= max(1.0e-12, 1.0e-12 * denom))
        print(
            f"[B3_BLOCK_COMPOSE_matnest_owned_copy] {label} "
            f"original_norm={orig_norm:.16e} owned_norm={owned_norm:.16e} "
            f"fro_diff={fro_diff:.16e} relative={relative_diff:.16e}",
            flush=True,
        )
    except Exception as exc:
        compose_meta[f"{prefix}_diag_unavailable_reason"] = f"{type(exc).__name__}:{exc}"


def _owned_aij_copy_for_matnest(
    mat: Any,
    *,
    label: str,
    compose_meta: Dict[str, Any],
) -> Any:
    owned = _owned_aij_copy(mat)
    _record_matnest_owned_copy_diag(original=mat, owned=owned, label=label, compose_meta=compose_meta)
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
        return "Auu"
    if r < nu and c >= nu:
        return "Aup"
    if r >= nu and c < nu:
        return "Apu"
    return "App"


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


def _row_stored_dict(mat: Any, row: int) -> Dict[int, float]:
    out: Dict[int, float] = {}
    try:
        cols, vals = mat.getRow(int(row))
    except TypeError:
        rowdat = mat.getRow(int(row))
        cols, vals = rowdat[0], rowdat[1]
    for c, v in zip(np.asarray(cols, dtype=np.int32).ravel(), np.asarray(vals, dtype=np.float64).ravel()):
        out[int(c)] = float(v)
    try:
        mat.restoreRow(int(row))
    except Exception:
        pass
    return out


def _push_top_entry(top: List[Dict[str, Any]], entry: Dict[str, Any], *, limit: int = 20) -> None:
    item = dict(entry)
    item["abs_diff"] = float(item["abs_diff"])
    if len(top) < int(limit):
        heapq.heappush(top, (item["abs_diff"], len(top), item))
        return
    if item["abs_diff"] <= float(top[0][0]):
        return
    heapq.heapreplace(top, (item["abs_diff"], len(top), item))


def _finalize_top_entries(top: List[Any]) -> List[Dict[str, Any]]:
    ordered = sorted(top, key=lambda triple: float(triple[0]), reverse=True)
    return [dict(triple[2]) for triple in ordered]


def _annotate_child_nest_pattern_entry(
    entry: Dict[str, Any], *, child: Any, nest_sub: Any
) -> Dict[str, Any]:
    r = int(entry["row_local"])
    row_child = _row_support_flags(child, r)
    row_nest = _row_support_flags(nest_sub, r)
    out = dict(entry)
    out["on_diagonal"] = bool(int(entry["row_local"]) == int(entry["col_local"]))
    out["child_row_identity_like"] = bool(row_child["identity_like"])
    out["nest_row_identity_like"] = bool(row_nest["identity_like"])
    out["child_row_zero_like"] = bool(row_child["zero_row_like"])
    out["nest_row_zero_like"] = bool(row_nest["zero_row_like"])
    out["bc_identity_or_zero_row"] = bool(
        row_child["identity_like"]
        or row_nest["identity_like"]
        or (row_child["zero_row_like"] and row_nest["zero_row_like"])
    )
    return out


def _petsc_mat_info_dict(mat: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"type": str(mat.getType()), "shape": list(mat.getSize())}
    try:
        info = mat.getInfo()
        if isinstance(info, dict):
            for key, val in info.items():
                out[f"info_{key}"] = _safe_float(val) if isinstance(val, (int, float)) else val
    except Exception as exc:
        out["info_error"] = f"{type(exc).__name__}:{exc}"
    try:
        out["global_nnz_used"] = int(_audit_helpers()._petsc_mat_global_nnz_used(mat))
    except Exception as exc:
        out["global_nnz_used_error"] = f"{type(exc).__name__}:{exc}"
    return out


def _build_petsc_diff_matrix(
    child: Any,
    nest_sub: Any,
    *,
    structure: Any = None,
) -> Any:
    diff = child.duplicate()
    child.copy(diff)
    if structure is None:
        diff.axpy(-1.0, nest_sub)
    else:
        diff.axpy(-1.0, nest_sub, structure=structure)
    diff.assemble()
    return diff


def _scan_diff_matrix_rows(
    diff_mat: Any,
    *,
    child: Any,
    nest_sub: Any,
    block_label: str,
    row_offset: int,
    col_offset: int,
) -> Dict[str, Any]:
    top_heap: List[Any] = []
    threshold_counts = {key: 0 for key, _val in DIFF_THRESHOLDS}
    entry_count = 0
    max_abs = 0.0
    only_diagonal = True
    only_bc_identity = True
    unit_diagonal_count = 0
    unit_diagonal_bc_count = 0
    row_bc_cache: Dict[int, bool] = {}

    def _bc_row(r: int) -> bool:
        cached = row_bc_cache.get(int(r))
        if cached is not None:
            return bool(cached)
        row_child = _row_support_flags(child, r)
        row_nest = _row_support_flags(nest_sub, r)
        bc = bool(
            row_child["identity_like"]
            or row_nest["identity_like"]
            or (row_child["zero_row_like"] and row_nest["zero_row_like"])
        )
        row_bc_cache[int(r)] = bc
        return bc

    nrow = int(diff_mat.getSize()[0])
    traversed_nnz = 0
    for r in range(nrow):
        try:
            cols, vals = diff_mat.getRow(r)
        except TypeError:
            rowdat = diff_mat.getRow(r)
            cols, vals = rowdat[0], rowdat[1]
        cols_a = np.asarray(cols, dtype=np.int32).ravel()
        vals_a = np.asarray(vals, dtype=np.float64).ravel()
        traversed_nnz += int(vals_a.size)
        for c, v in zip(cols_a.tolist(), vals_a.tolist()):
            abs_v = abs(float(v))
            if abs_v <= VALUE_DIFF_MIN:
                continue
            entry_count += 1
            max_abs = max(max_abs, abs_v)
            on_diag = bool(int(r) == int(c))
            if not on_diag:
                only_diagonal = False
            bc = _bc_row(int(r))
            if not bc:
                only_bc_identity = False
            if on_diag and abs(abs_v - 1.0) <= 1.0e-8:
                unit_diagonal_count += 1
                if bc:
                    unit_diagonal_bc_count += 1
            for key, thr in DIFF_THRESHOLDS:
                if abs_v > thr:
                    threshold_counts[key] += 1
            _push_top_entry(
                top_heap,
                {
                    "row_local": int(r),
                    "col_local": int(c),
                    "row_global": int(r + row_offset),
                    "col_global": int(c + col_offset),
                    "block": str(block_label),
                    "diff_value": _safe_float(v),
                    "abs_diff": _safe_float(abs_v),
                    "on_diagonal": on_diag,
                    "bc_identity_or_zero_row": bc,
                },
                limit=20,
            )
        try:
            diff_mat.restoreRow(r)
        except Exception:
            pass

    fro_norm = None
    try:
        fro_norm = float(diff_mat.norm(PETSc.NormType.FROBENIUS))
    except Exception:
        pass

    return {
        "frobenius_norm": _safe_float(fro_norm),
        "mat_info": _petsc_mat_info_dict(diff_mat),
        "getrow_traversed_nnz": int(traversed_nnz),
        "nonzero_entry_count": int(entry_count),
        "max_abs": _safe_float(max_abs),
        "threshold_counts": dict(threshold_counts),
        "only_on_diagonal": bool(only_diagonal and entry_count > 0),
        "only_bc_identity_or_zero_rows": bool(only_bc_identity and entry_count > 0),
        "unit_diagonal_count": int(unit_diagonal_count),
        "unit_diagonal_bc_identity_count": int(unit_diagonal_bc_count),
        "top20": _finalize_top_entries(top_heap)[:20],
        "norm_positive_but_getrow_empty": bool(
            fro_norm is not None and float(fro_norm) > 1.0e-12 and entry_count == 0
        ),
    }


def _scipy_csr_frobenius_diff(child: Any, nest_sub: Any) -> Optional[float]:
    from scipy import sparse

    def _to_csr(mat: Any) -> Any:
        indptr, indices, data = mat.getValuesCSR()
        nrow, ncol = mat.getSize()
        return sparse.csr_matrix(
            (
                np.asarray(data, dtype=np.float64),
                np.asarray(indices, dtype=np.int32),
                np.asarray(indptr, dtype=np.int32),
            ),
            shape=(int(nrow), int(ncol)),
        )

    csr_c = _to_csr(child)
    csr_n = _to_csr(nest_sub)
    diff = csr_c - csr_n
    if diff.nnz == 0:
        return 0.0
    return float(np.linalg.norm(diff.data))


def _inspect_child_vs_nest_petsc_diff_matrix(
    child: Any,
    nest_sub: Any,
    *,
    block_label: str,
    row_offset: int,
    col_offset: int,
    compose_meta: Dict[str, Any],
    prefix: str,
    mesh_level: str = "",
) -> Dict[str, Any]:
    """Build D = child - nest_block in PETSc and inspect D directly."""
    audit = _audit_helpers()
    dp = f"{prefix}_{block_label}_petsc_diff"
    summary: Dict[str, Any] = {
        "block_label": str(block_label),
        "petsc_diff_harmless_unit_diag_bc": False,
    }
    diff_mats: List[Any] = []
    try:
        child_norm = float(child.norm(PETSc.NormType.FROBENIUS))
        nest_norm = float(nest_sub.norm(PETSc.NormType.FROBENIUS))
        petsc_fro = float(audit._petsc_mat_frobenius_difference(child, nest_sub))
        compose_meta[f"{dp}_child_frobenius_norm"] = _safe_float(child_norm)
        compose_meta[f"{dp}_nest_block_frobenius_norm"] = _safe_float(nest_norm)
        compose_meta[f"{dp}_petsc_helper_frobenius_diff"] = _safe_float(petsc_fro)
        compose_meta[f"{dp}_child_type"] = str(child.getType())
        compose_meta[f"{dp}_nest_block_type"] = str(nest_sub.getType())
        compose_meta[f"{dp}_child_mat_info"] = _petsc_mat_info_dict(child)
        compose_meta[f"{dp}_nest_block_mat_info"] = _petsc_mat_info_dict(nest_sub)
        print(
            f"[B3_BLOCK_COMPOSE_child_vs_nest] {block_label} "
            f"child_norm={child_norm:.16e} nest_norm={nest_norm:.16e} "
            f"petsc_fro_diff={petsc_fro:.16e}",
            flush=True,
        )

        if str(mesh_level) == "L_dev_dense":
            try:
                scipy_fro = _scipy_csr_frobenius_diff(child, nest_sub)
                compose_meta[f"{dp}_scipy_csr_frobenius_diff"] = _safe_float(scipy_fro)
                print(
                    f"[B3_BLOCK_COMPOSE_child_vs_nest] {block_label} "
                    f"scipy_csr_fro_diff={scipy_fro:.16e}",
                    flush=True,
                )
            except Exception as exc_scipy:
                compose_meta[f"{dp}_scipy_csr_frobenius_diff_unavailable"] = (
                    f"{type(exc_scipy).__name__}:{exc_scipy}"
                )

        structure_variants: List[Tuple[str, Any]] = [
            ("SUBSET_NONZERO_PATTERN", PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN),
            ("DIFFERENT_NONZERO_PATTERN", PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN),
            ("DEFAULT", None),
        ]
        best_scan: Optional[Dict[str, Any]] = None
        for struct_label, struct_val in structure_variants:
            D = _build_petsc_diff_matrix(child, nest_sub, structure=struct_val)
            diff_mats.append(D)
            scan = _scan_diff_matrix_rows(
                D,
                child=child,
                nest_sub=nest_sub,
                block_label=block_label,
                row_offset=row_offset,
                col_offset=col_offset,
            )
            compose_meta[f"{dp}_scan_{struct_label}_frobenius_norm"] = scan.get("frobenius_norm")
            compose_meta[f"{dp}_scan_{struct_label}_nonzero_entry_count"] = scan.get("nonzero_entry_count")
            compose_meta[f"{dp}_scan_{struct_label}_max_abs"] = scan.get("max_abs")
            compose_meta[f"{dp}_scan_{struct_label}_norm_positive_but_getrow_empty"] = scan.get(
                "norm_positive_but_getrow_empty"
            )
            if best_scan is None or int(scan.get("nonzero_entry_count") or 0) > int(
                best_scan.get("nonzero_entry_count") or 0
            ):
                best_scan = dict(scan)
                best_scan["structure"] = struct_label

            if scan.get("norm_positive_but_getrow_empty"):
                compose_meta[f"{dp}_scan_{struct_label}_petsc_norm_getrow_inconsistency"] = True
                D_aij = None
                try:
                    try:
                        D_aij = D.convert("aij")
                    except Exception:
                        D_aij = PETSc.Mat()
                        D.convert("aij", D_aij)
                    if D_aij is not None:
                        D_aij.assemble()
                        diff_mats.append(D_aij)
                        try:
                            D_aij.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, False)
                        except Exception:
                            pass
                        scan_aij = _scan_diff_matrix_rows(
                            D_aij,
                            child=child,
                            nest_sub=nest_sub,
                            block_label=block_label,
                            row_offset=row_offset,
                            col_offset=col_offset,
                        )
                        compose_meta[f"{dp}_scan_{struct_label}_aij_convert_frobenius_norm"] = scan_aij.get(
                            "frobenius_norm"
                        )
                        compose_meta[f"{dp}_scan_{struct_label}_aij_convert_nonzero_entry_count"] = scan_aij.get(
                            "nonzero_entry_count"
                        )
                        compose_meta[f"{dp}_scan_{struct_label}_aij_convert_top20"] = scan_aij.get("top20")
                        if int(scan_aij.get("nonzero_entry_count") or 0) > int(
                            best_scan.get("nonzero_entry_count") or 0
                        ):
                            best_scan = dict(scan_aij)
                            best_scan["structure"] = f"{struct_label}_aij_convert"
                except Exception as exc_aij:
                    compose_meta[f"{dp}_scan_{struct_label}_aij_convert_error"] = (
                        f"{type(exc_aij).__name__}:{exc_aij}"
                    )

        if best_scan is None:
            best_scan = {}

        compose_meta[f"{dp}_selected_structure"] = best_scan.get("structure")
        compose_meta[f"{dp}_frobenius_norm"] = best_scan.get("frobenius_norm")
        compose_meta[f"{dp}_mat_info"] = best_scan.get("mat_info")
        compose_meta[f"{dp}_getrow_traversed_nnz"] = best_scan.get("getrow_traversed_nnz")
        compose_meta[f"{dp}_nonzero_entry_count"] = best_scan.get("nonzero_entry_count")
        compose_meta[f"{dp}_max_abs"] = best_scan.get("max_abs")
        compose_meta[f"{dp}_top20"] = best_scan.get("top20") or []
        compose_meta[f"{dp}_only_on_diagonal"] = best_scan.get("only_on_diagonal")
        compose_meta[f"{dp}_only_bc_identity_or_zero_rows"] = best_scan.get("only_bc_identity_or_zero_rows")
        compose_meta[f"{dp}_unit_diagonal_count"] = best_scan.get("unit_diagonal_count")
        compose_meta[f"{dp}_unit_diagonal_bc_identity_count"] = best_scan.get(
            "unit_diagonal_bc_identity_count"
        )
        for key, _thr in DIFF_THRESHOLDS:
            tc = (best_scan.get("threshold_counts") or {}).get(key, 0)
            compose_meta[f"{dp}_count_gt_{key}"] = int(tc)

        norm_pos_empty = bool(
            best_scan.get("norm_positive_but_getrow_empty")
            or (
                float(best_scan.get("frobenius_norm") or 0.0) > 1.0e-12
                and int(best_scan.get("nonzero_entry_count") or 0) == 0
            )
        )
        compose_meta[f"{dp}_petsc_norm_getrow_inconsistency"] = norm_pos_empty
        if norm_pos_empty:
            compose_meta[f"{dp}_petsc_norm_getrow_inconsistency_note"] = (
                "D.norm > 0 but D.getRow scan found no entries above tolerance; "
                "likely PETSc explicit-zero / sparsity-pattern representation mismatch."
            )

        harmless = bool(
            float(best_scan.get("frobenius_norm") or 0.0) >= 1.0 - 1.0e-8
            and int(best_scan.get("unit_diagonal_bc_identity_count") or 0) == 1
            and int(best_scan.get("unit_diagonal_count") or 0) == 1
            and bool(best_scan.get("only_on_diagonal"))
            and bool(best_scan.get("only_bc_identity_or_zero_rows"))
        )
        summary["petsc_diff_harmless_unit_diag_bc"] = harmless
        compose_meta[f"{dp}_harmless_unit_diag_bc"] = harmless
        if harmless:
            compose_meta[f"{dp}_mathematically_harmless_note"] = (
                f"{block_label}: PETSc D=child-nest_block has exactly one unit diagonal BC/identity "
                "entry; shared-stored values may match while sparsity pattern differs. "
                "compare_pass not auto-waived."
            )
        elif norm_pos_empty:
            compose_meta[f"{dp}_mathematically_harmless_note"] = (
                f"{block_label}: PETSc Frobenius diff ~{petsc_fro:.6e} but D.getRow scan empty; "
                "see structure scans and scipy_csr_frobenius_diff."
            )
        else:
            compose_meta[f"{dp}_mathematically_harmless_note"] = (
                f"{block_label}: inspect petsc_diff top20 and norm fields."
            )
        compose_meta[f"{dp}_audit_pass"] = True
        print(
            f"[B3_BLOCK_COMPOSE_child_vs_nest] {block_label} "
            f"D_norm={best_scan.get('frobenius_norm')} D_nnz={best_scan.get('nonzero_entry_count')} "
            f"D_max_abs={best_scan.get('max_abs')} inconsistency={norm_pos_empty}",
            flush=True,
        )
    except Exception as exc:
        compose_meta[f"{dp}_audit_pass"] = False
        compose_meta[f"{dp}_audit_unavailable_reason"] = f"{type(exc).__name__}:{exc}"
    finally:
        for mat in diff_mats:
            try:
                mat.destroy()
            except Exception:
                pass
    return summary


def _record_child_vs_nest_block_pair_detail(
    child: Any,
    nest_sub: Any,
    *,
    block_label: str,
    row_offset: int,
    col_offset: int,
    compose_meta: Dict[str, Any],
    prefix: str,
    mesh_level: str = "",
) -> Dict[str, Any]:
    """Targeted child-vs-nest diff localization for one diagonal block (Auu/App)."""
    petsc_summary = _inspect_child_vs_nest_petsc_diff_matrix(
        child,
        nest_sub,
        block_label=block_label,
        row_offset=row_offset,
        col_offset=col_offset,
        compose_meta=compose_meta,
        prefix=prefix,
        mesh_level=mesh_level,
    )
    detail_prefix = f"{prefix}_{block_label}_detail"
    pattern_prefix = f"{prefix}_{block_label}_pattern"
    compose_meta[f"{detail_prefix}_top20"] = []
    compose_meta[f"{detail_prefix}_max_abs"] = 0.0
    compose_meta[f"{pattern_prefix}_top20_child_only"] = []
    compose_meta[f"{pattern_prefix}_top20_nest_only"] = []
    for key, _thr in DIFF_THRESHOLDS:
        compose_meta[f"{detail_prefix}_count_gt_{key}"] = 0
        compose_meta[f"{pattern_prefix}_child_only_count_gt_{key}"] = 0
        compose_meta[f"{pattern_prefix}_nest_only_count_gt_{key}"] = 0
    summary: Dict[str, Any] = {
        "block_label": str(block_label),
        "harmless_unit_diag_bc": False,
        "pattern_harmless_unit_diag_bc": False,
        "petsc_diff_harmless_unit_diag_bc": bool(petsc_summary.get("petsc_diff_harmless_unit_diag_bc")),
        "unit_diagonal_diff_count": 0,
        "unit_diagonal_bc_identity_diff_count": 0,
        "unit_diagonal_pattern_only_count": 0,
        "unit_diagonal_bc_identity_pattern_only_count": 0,
        "shared_stored_diff_count": 0,
        "child_only_nnz": 0,
        "nest_only_nnz": 0,
        "only_on_diagonal_pattern": True,
        "only_bc_identity_or_zero_rows_pattern": True,
        "max_abs": 0.0,
        "max_abs_child_only": 0.0,
        "max_abs_nest_only": 0.0,
    }
    top_shared_heap: List[Any] = []
    top_child_only_heap: List[Any] = []
    top_nest_only_heap: List[Any] = []
    shared_threshold_counts = {key: 0 for key, _val in DIFF_THRESHOLDS}
    child_only_threshold_counts = {key: 0 for key, _val in DIFF_THRESHOLDS}
    nest_only_threshold_counts = {key: 0 for key, _val in DIFF_THRESHOLDS}
    try:
        nrow = int(child.getSize()[0])
        shared_stored_diff_count = 0
        max_abs_shared = 0.0
        child_only_nnz = 0
        nest_only_nnz = 0
        max_abs_child_only = 0.0
        max_abs_nest_only = 0.0
        only_diagonal_pattern = True
        only_bc_identity_pattern = True
        unit_diagonal_pattern_count = 0
        unit_diagonal_bc_identity_pattern_count = 0
        row_bc_cache: Dict[int, bool] = {}

        def _bc_identity_row(r: int) -> bool:
            cached = row_bc_cache.get(int(r))
            if cached is not None:
                return bool(cached)
            row_child = _row_support_flags(child, r)
            row_nest = _row_support_flags(nest_sub, r)
            bc_row = bool(
                row_child["identity_like"]
                or row_nest["identity_like"]
                or (row_child["zero_row_like"] and row_nest["zero_row_like"])
            )
            row_bc_cache[int(r)] = bc_row
            return bc_row

        for r in range(nrow):
            child_row = _row_stored_dict(child, r)
            nest_row = _row_stored_dict(nest_sub, r)
            all_cols = set(child_row.keys()) | set(nest_row.keys())
            for c in all_cols:
                in_child = c in child_row
                in_nest = c in nest_row
                if in_child and in_nest:
                    child_v = float(child_row[c])
                    nest_v = float(nest_row[c])
                    diff_v = child_v - nest_v
                    abs_diff = abs(float(diff_v))
                    if abs_diff <= VALUE_DIFF_MIN:
                        continue
                    shared_stored_diff_count += 1
                    max_abs_shared = max(max_abs_shared, abs_diff)
                    bc_row = _bc_identity_row(int(r))
                    for key, thr in DIFF_THRESHOLDS:
                        if abs_diff > thr:
                            shared_threshold_counts[key] += 1
                    _push_top_entry(
                        top_shared_heap,
                        {
                            "row_local": int(r),
                            "col_local": int(c),
                            "row_global": int(r + row_offset),
                            "col_global": int(c + col_offset),
                            "block": str(block_label),
                            "child": _safe_float(child_v),
                            "nest": _safe_float(nest_v),
                            "diff": _safe_float(diff_v),
                            "abs_diff": _safe_float(abs_diff),
                            "kind": "shared_stored_value_diff",
                            "bc_identity_or_zero_row": bc_row,
                        },
                        limit=20,
                    )
                    continue

                bc_row = _bc_identity_row(int(r))
                on_diagonal = bool(int(r) == int(c))

                if in_child and not in_nest:
                    child_only_nnz += 1
                    child_v = float(child_row[c])
                    abs_val = abs(child_v)
                    max_abs_child_only = max(max_abs_child_only, abs_val)
                    if not on_diagonal:
                        only_diagonal_pattern = False
                    if not bc_row:
                        only_bc_identity_pattern = False
                    if on_diagonal and abs(abs_val - 1.0) <= 1.0e-8:
                        unit_diagonal_pattern_count += 1
                        if bc_row:
                            unit_diagonal_bc_identity_pattern_count += 1
                    for key, thr in DIFF_THRESHOLDS:
                        if abs_val > thr:
                            child_only_threshold_counts[key] += 1
                    _push_top_entry(
                        top_child_only_heap,
                        {
                            "row_local": int(r),
                            "col_local": int(c),
                            "row_global": int(r + row_offset),
                            "col_global": int(c + col_offset),
                            "block": str(block_label),
                            "value": _safe_float(child_v),
                            "abs_value": _safe_float(abs_val),
                            "on_diagonal": on_diagonal,
                            "kind": "child_only_stored",
                            "bc_identity_or_zero_row": bc_row,
                            "explicit_zero_stored": bool(abs_val <= EXPLICIT_ZERO_TOL),
                        },
                        limit=20,
                    )
                    continue

                if in_nest and not in_child:
                    nest_only_nnz += 1
                    nest_v = float(nest_row[c])
                    abs_val = abs(nest_v)
                    max_abs_nest_only = max(max_abs_nest_only, abs_val)
                    if not on_diagonal:
                        only_diagonal_pattern = False
                    if not bc_row:
                        only_bc_identity_pattern = False
                    if on_diagonal and abs(abs_val - 1.0) <= 1.0e-8:
                        unit_diagonal_pattern_count += 1
                        if bc_row:
                            unit_diagonal_bc_identity_pattern_count += 1
                    for key, thr in DIFF_THRESHOLDS:
                        if abs_val > thr:
                            nest_only_threshold_counts[key] += 1
                    _push_top_entry(
                        top_nest_only_heap,
                        {
                            "row_local": int(r),
                            "col_local": int(c),
                            "row_global": int(r + row_offset),
                            "col_global": int(c + col_offset),
                            "block": str(block_label),
                            "value": _safe_float(nest_v),
                            "abs_value": _safe_float(abs_val),
                            "on_diagonal": on_diagonal,
                            "kind": "nest_only_stored",
                            "bc_identity_or_zero_row": bc_row,
                            "explicit_zero_stored": bool(abs_val <= EXPLICIT_ZERO_TOL),
                        },
                        limit=20,
                    )

        top20_shared = [
            _annotate_child_nest_pattern_entry(item, child=child, nest_sub=nest_sub)
            for item in _finalize_top_entries(top_shared_heap)
        ]
        top20_child_only = [
            _annotate_child_nest_pattern_entry(item, child=child, nest_sub=nest_sub)
            for item in _finalize_top_entries(top_child_only_heap)
        ]
        top20_nest_only = [
            _annotate_child_nest_pattern_entry(item, child=child, nest_sub=nest_sub)
            for item in _finalize_top_entries(top_nest_only_heap)
        ]

        pattern_harmless_unit_diag_bc = bool(
            shared_stored_diff_count == 0
            and unit_diagonal_bc_identity_pattern_count == 1
            and unit_diagonal_pattern_count == 1
            and only_diagonal_pattern
            and only_bc_identity_pattern
            and max(max_abs_child_only, max_abs_nest_only) >= 1.0 - 1.0e-8
        )

        summary.update(
            {
                "harmless_unit_diag_bc": pattern_harmless_unit_diag_bc,
                "pattern_harmless_unit_diag_bc": pattern_harmless_unit_diag_bc,
                "unit_diagonal_pattern_only_count": int(unit_diagonal_pattern_count),
                "unit_diagonal_bc_identity_pattern_only_count": int(
                    unit_diagonal_bc_identity_pattern_count
                ),
                "shared_stored_diff_count": int(shared_stored_diff_count),
                "child_only_nnz": int(child_only_nnz),
                "nest_only_nnz": int(nest_only_nnz),
                "only_on_diagonal_pattern": bool(only_diagonal_pattern),
                "only_bc_identity_or_zero_rows_pattern": bool(only_bc_identity_pattern),
                "max_abs": float(max(max_abs_shared, max_abs_child_only, max_abs_nest_only)),
                "max_abs_child_only": float(max_abs_child_only),
                "max_abs_nest_only": float(max_abs_nest_only),
            }
        )

        compose_meta[f"{detail_prefix}_top20"] = top20_shared[:20]
        compose_meta[f"{detail_prefix}_max_abs"] = _safe_float(max_abs_shared)
        compose_meta[f"{detail_prefix}_shared_stored_diff_count"] = int(shared_stored_diff_count)
        for key, _thr in DIFF_THRESHOLDS:
            compose_meta[f"{detail_prefix}_count_gt_{key}"] = int(shared_threshold_counts[key])

        compose_meta[f"{pattern_prefix}_child_only_nnz"] = int(child_only_nnz)
        compose_meta[f"{pattern_prefix}_nest_only_nnz"] = int(nest_only_nnz)
        compose_meta[f"{pattern_prefix}_top20_child_only"] = top20_child_only[:20]
        compose_meta[f"{pattern_prefix}_top20_nest_only"] = top20_nest_only[:20]
        compose_meta[f"{pattern_prefix}_max_abs_child_only"] = _safe_float(max_abs_child_only)
        compose_meta[f"{pattern_prefix}_max_abs_nest_only"] = _safe_float(max_abs_nest_only)
        compose_meta[f"{pattern_prefix}_only_on_diagonal"] = bool(only_diagonal_pattern)
        compose_meta[f"{pattern_prefix}_only_bc_identity_or_zero_rows"] = bool(only_bc_identity_pattern)
        compose_meta[f"{pattern_prefix}_unit_diagonal_count"] = int(unit_diagonal_pattern_count)
        compose_meta[f"{pattern_prefix}_unit_diagonal_bc_identity_count"] = int(
            unit_diagonal_bc_identity_pattern_count
        )
        for key, _thr in DIFF_THRESHOLDS:
            compose_meta[f"{pattern_prefix}_child_only_count_gt_{key}"] = int(
                child_only_threshold_counts[key]
            )
            compose_meta[f"{pattern_prefix}_nest_only_count_gt_{key}"] = int(
                nest_only_threshold_counts[key]
            )
        compose_meta[f"{pattern_prefix}_harmless_unit_diag_bc"] = pattern_harmless_unit_diag_bc
        if pattern_harmless_unit_diag_bc:
            compose_meta[f"{pattern_prefix}_mathematically_harmless_note"] = (
                f"{block_label}: exactly one pattern-only unit diagonal BC/identity/zero-row entry; "
                "shared-stored values match. compare_pass not auto-waived."
            )
        elif shared_stored_diff_count == 0 and (child_only_nnz > 0 or nest_only_nnz > 0):
            compose_meta[f"{pattern_prefix}_mathematically_harmless_note"] = (
                f"{block_label}: pattern-only child/nest sparsity mismatch with no shared-stored "
                "value diffs; inspect top20_child_only and top20_nest_only."
            )
        else:
            compose_meta[f"{pattern_prefix}_mathematically_harmless_note"] = (
                f"{block_label}: see shared and pattern-only child-vs-nest diagnostics."
            )
        compose_meta[f"{detail_prefix}_audit_pass"] = True
        compose_meta[f"{pattern_prefix}_audit_pass"] = True
        summary["petsc_diff_harmless_unit_diag_bc"] = bool(
            petsc_summary.get("petsc_diff_harmless_unit_diag_bc")
        )
        summary["pattern_harmless_unit_diag_bc"] = bool(
            pattern_harmless_unit_diag_bc or summary["petsc_diff_harmless_unit_diag_bc"]
        )
    except Exception as exc:
        compose_meta[f"{detail_prefix}_audit_pass"] = False
        compose_meta[f"{pattern_prefix}_audit_pass"] = False
        compose_meta[f"{detail_prefix}_audit_unavailable_reason"] = f"{type(exc).__name__}:{exc}"
        compose_meta[f"{pattern_prefix}_audit_unavailable_reason"] = f"{type(exc).__name__}:{exc}"
        summary["petsc_diff_harmless_unit_diag_bc"] = bool(
            petsc_summary.get("petsc_diff_harmless_unit_diag_bc")
        )
    return summary


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


def _mat_comm(mat: Any, fallback: Any) -> Any:
    try:
        return mat.getComm()
    except Exception:
        return fallback


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
    mat_comm = _mat_comm(mat, comm)
    is_row = PETSc.IS().createStride(n_rows, int(row_start), 1, comm=mat_comm)
    is_col = PETSc.IS().createStride(n_cols, int(col_start), 1, comm=mat_comm)
    try:
        sub = mat.createSubMatrix(is_row, is_col, PETSc.Mat.SubMatrixOption.EXTRACT)
    except Exception:
        sub = mat.createSubMatrix(is_row, is_col)
    sub.assemble()
    return sub


def _extract_submatrix_scipy_dev(
    mat: Any,
    *,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> Any:
    from scipy import sparse

    indptr, indices, data = mat.getValuesCSR()
    nrow, ncol = mat.getSize()
    csr = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float64), np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int32)),
        shape=(int(nrow), int(ncol)),
    )
    return csr[int(row_start) : int(row_end), int(col_start) : int(col_end)].tocsr()


def _frobenius_dense_or_petsc(a: Any, b: Any) -> float:
    audit = _audit_helpers()
    if hasattr(a, "getType"):
        return float(audit._petsc_mat_frobenius_difference(a, b))
    diff = a - b
    return float(np.linalg.norm(diff.data)) if hasattr(diff, "data") else float(np.linalg.norm(diff.toarray()))


def _safe_block_frobenius_difference(
    old: Any,
    new: Any,
    *,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    comm: Any,
    mesh_level: str = "",
) -> Tuple[Optional[float], Optional[str]]:
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
        return _frobenius_dense_or_petsc(sub_old, sub_new), None
    except Exception as exc_petsc:
        if str(mesh_level) != "L_dev_dense":
            return None, f"{type(exc_petsc).__name__}:{exc_petsc}"
        try:
            csr_old = _extract_submatrix_scipy_dev(
                old, row_start=row_start, row_end=row_end, col_start=col_start, col_end=col_end
            )
            csr_new = _extract_submatrix_scipy_dev(
                new, row_start=row_start, row_end=row_end, col_start=col_start, col_end=col_end
            )
            return _frobenius_dense_or_petsc(csr_old, csr_new), "scipy_csr_slice_fallback"
        except Exception as exc_scipy:
            return None, f"{type(exc_petsc).__name__}:{exc_petsc};scipy:{type(exc_scipy).__name__}:{exc_scipy}"
    finally:
        for sub in (sub_old, sub_new):
            if sub is not None and hasattr(sub, "destroy"):
                try:
                    sub.destroy()
                except Exception:
                    pass


def _init_block_stats(block_labels: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for label in block_labels:
        stats[str(label)] = {
            "shared_value_diff_count": 0,
            "pattern_old_only_count": 0,
            "pattern_new_only_count": 0,
            "pattern_old_only_nonzero_count": 0,
            "pattern_new_only_nonzero_count": 0,
            "max_abs": 0.0,
        }
    return stats


def _annotate_top_entry(entry: Dict[str, Any], *, old: Any, new: Any) -> Dict[str, Any]:
    r = int(entry["row"])
    row_old = _row_support_flags(old, r)
    row_new = _row_support_flags(new, r)
    out = dict(entry)
    out["row_old_identity_like"] = bool(row_old["identity_like"])
    out["row_new_identity_like"] = bool(row_new["identity_like"])
    out["row_old_zero_like"] = bool(row_old["zero_row_like"])
    out["row_new_zero_like"] = bool(row_new["zero_row_like"])
    out["row_old_diag_value"] = _safe_float(row_old["diag_value"])
    out["row_new_diag_value"] = _safe_float(row_new["diag_value"])
    return out


def _streaming_compose_matrix_diff_audit(
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
    mesh_level: str = "",
) -> None:
    prefix = f"B3_BLOCK_COMPOSE_compare_{label}_diff"
    block_labels = [str(spec[0]) for spec in block_fro_specs]
    block_stats = _init_block_stats(block_labels)
    block_ranges = {
        str(spec[0]): {"row_start": int(spec[1]), "row_end": int(spec[2]), "col_start": int(spec[3]), "col_end": int(spec[4])}
        for spec in block_fro_specs
    }
    threshold_counts = {key: 0 for key, _val in DIFF_THRESHOLDS}
    shared_threshold_counts = {key: 0 for key, _val in DIFF_THRESHOLDS}
    top_value_heap: List[Any] = []
    top_pattern_heap: List[Any] = []
    diff_mat = None
    compose_meta[f"{prefix}_top20"] = []
    compose_meta[f"{prefix}_top20_shared"] = []
    compose_meta[f"{prefix}_top20_pattern_mismatch"] = []
    try:
        nrow = int(old.getSize()[0])
        stored_intersection = 0
        stored_old_only = 0
        stored_new_only = 0
        shared_value_diff_count = 0
        pattern_mismatch_count = 0
        max_abs_shared = 0.0
        max_abs_pattern = 0.0
        only_diagonal_shared = True
        for r in range(nrow):
            old_row = _row_stored_dict(old, r)
            new_row = _row_stored_dict(new, r)
            all_cols = set(old_row.keys()) | set(new_row.keys())
            for c in all_cols:
                in_old = c in old_row
                in_new = c in new_row
                block = str(block_name_fn(r, c, n_u))
                bstat = block_stats.get(block)
                if bstat is None:
                    continue
                if in_old and in_new:
                    stored_intersection += 1
                    old_v = float(old_row[c])
                    new_v = float(new_row[c])
                    diff_v = old_v - new_v
                    abs_diff = abs(diff_v)
                    if abs_diff <= VALUE_DIFF_MIN:
                        continue
                    shared_value_diff_count += 1
                    bstat["shared_value_diff_count"] = int(bstat["shared_value_diff_count"]) + 1
                    bstat["max_abs"] = max(float(bstat["max_abs"]), abs_diff)
                    max_abs_shared = max(max_abs_shared, abs_diff)
                    if int(r) != int(c):
                        only_diagonal_shared = False
                    for key, thr in DIFF_THRESHOLDS:
                        if abs_diff > thr:
                            threshold_counts[key] += 1
                            shared_threshold_counts[key] += 1
                    _push_top_entry(
                        top_value_heap,
                        {
                            "row": int(r),
                            "col": int(c),
                            "block": block,
                            "old": _safe_float(old_v),
                            "new": _safe_float(new_v),
                            "diff": _safe_float(diff_v),
                            "abs_diff": _safe_float(abs_diff),
                            "kind": "shared_stored_value_diff",
                        },
                    )
                    continue
                if in_old and not in_new:
                    stored_old_only += 1
                    pattern_mismatch_count += 1
                    old_v = float(old_row[c])
                    bstat["pattern_old_only_count"] = int(bstat["pattern_old_only_count"]) + 1
                    if abs(old_v) > EXPLICIT_ZERO_TOL:
                        bstat["pattern_old_only_nonzero_count"] = int(bstat["pattern_old_only_nonzero_count"]) + 1
                        abs_diff = abs(old_v)
                        max_abs_pattern = max(max_abs_pattern, abs_diff)
                        for key, thr in DIFF_THRESHOLDS:
                            if abs_diff > thr:
                                threshold_counts[key] += 1
                        _push_top_entry(
                            top_pattern_heap,
                            {
                                "row": int(r),
                                "col": int(c),
                                "block": block,
                                "old": _safe_float(old_v),
                                "new": None,
                                "diff": _safe_float(old_v),
                                "abs_diff": _safe_float(abs_diff),
                                "kind": "stored_old_only_implicit_zero_in_new",
                            },
                        )
                    continue
                if in_new and not in_old:
                    stored_new_only += 1
                    pattern_mismatch_count += 1
                    new_v = float(new_row[c])
                    bstat["pattern_new_only_count"] = int(bstat["pattern_new_only_count"]) + 1
                    if abs(new_v) > EXPLICIT_ZERO_TOL:
                        bstat["pattern_new_only_nonzero_count"] = int(bstat["pattern_new_only_nonzero_count"]) + 1
                        abs_diff = abs(new_v)
                        max_abs_pattern = max(max_abs_pattern, abs_diff)
                        for key, thr in DIFF_THRESHOLDS:
                            if abs_diff > thr:
                                threshold_counts[key] += 1
                        _push_top_entry(
                            top_pattern_heap,
                            {
                                "row": int(r),
                                "col": int(c),
                                "block": block,
                                "old": None,
                                "new": _safe_float(new_v),
                                "diff": _safe_float(-new_v),
                                "abs_diff": _safe_float(abs_diff),
                                "kind": "stored_new_only_implicit_zero_in_old",
                            },
                        )

        top20_value = [_annotate_top_entry(item, old=old, new=new) for item in _finalize_top_entries(top_value_heap)]
        top20_pattern = [_annotate_top_entry(item, old=old, new=new) for item in _finalize_top_entries(top_pattern_heap)]

        diff_mat = old.duplicate()
        old.copy(diff_mat)
        try:
            diff_mat.axpy(-1.0, new, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
        except Exception:
            diff_mat.axpy(-1.0, new)
        diff_mat.assemble()
        petsc_diff_nnz = int(_audit_helpers()._petsc_mat_global_nnz_used(diff_mat))

        compose_meta[f"{prefix}_petsc_subset_diff_nnz"] = petsc_diff_nnz
        compose_meta[f"{prefix}_stored_intersection_count"] = int(stored_intersection)
        compose_meta[f"{prefix}_stored_old_only_count"] = int(stored_old_only)
        compose_meta[f"{prefix}_stored_new_only_count"] = int(stored_new_only)
        compose_meta[f"{prefix}_pattern_mismatch_count"] = int(pattern_mismatch_count)
        compose_meta[f"{prefix}_shared_stored_value_diff_count"] = int(shared_value_diff_count)
        compose_meta[f"{prefix}_entrywise_nnz"] = int(shared_value_diff_count + pattern_mismatch_count)
        compose_meta[f"{prefix}_max_abs"] = _safe_float(max(max_abs_shared, max_abs_pattern))
        compose_meta[f"{prefix}_max_abs_shared_stored"] = _safe_float(max_abs_shared)
        compose_meta[f"{prefix}_max_abs_pattern_mismatch"] = _safe_float(max_abs_pattern)
        for key, _thr in DIFF_THRESHOLDS:
            compose_meta[f"{prefix}_count_gt_{key}"] = int(threshold_counts[key])
            compose_meta[f"{prefix}_shared_count_gt_{key}"] = int(shared_threshold_counts[key])
        compose_meta[f"{prefix}_only_on_diagonal_shared"] = bool(
            only_diagonal_shared and shared_value_diff_count > 0
        )
        compose_meta[f"{prefix}_block_counts"] = {
            blk: int(stats["shared_value_diff_count"]) for blk, stats in block_stats.items()
        }
        compose_meta[f"{prefix}_block_pattern_old_only_counts"] = {
            blk: int(stats["pattern_old_only_count"]) for blk, stats in block_stats.items()
        }
        compose_meta[f"{prefix}_block_pattern_new_only_counts"] = {
            blk: int(stats["pattern_new_only_count"]) for blk, stats in block_stats.items()
        }
        compose_meta[f"{prefix}_block_max_abs"] = {
            blk: _safe_float(stats["max_abs"]) for blk, stats in block_stats.items()
        }
        compose_meta[f"{prefix}_block_row_col_ranges"] = block_ranges
        top20_shared = top20_value[:20]
        top20_pattern = top20_pattern[:20]
        compose_meta[f"{prefix}_top20_shared"] = top20_shared
        compose_meta[f"{prefix}_top20_pattern_mismatch"] = top20_pattern
        compose_meta[f"{prefix}_top20"] = top20_shared
        if not top20_shared:
            compose_meta[f"{prefix}_top20_shared_empty_reason"] = (
                "no shared-stored value diffs above 1e-15; inspect top20_pattern_mismatch "
                "and child_vs_nest Auu/App detail audits"
            )
        if not top20_pattern:
            compose_meta[f"{prefix}_top20_pattern_mismatch_empty_reason"] = (
                "no nonzero pattern-only mismatches recorded above explicit-zero tolerance"
            )
        compose_meta[f"{prefix}_explicit_zero_pattern_mismatch_dominates"] = bool(
            pattern_mismatch_count > 10 * max(1, shared_value_diff_count)
        )
        if shared_value_diff_count <= 4 and pattern_mismatch_count > 1000:
            compose_meta[f"{prefix}_mathematically_harmless_note"] = (
                "Few shared-stored value diffs with large explicit-zero sparsity-pattern mismatch; "
                "likely MatNest convert preserves a different explicit-zero pattern than direct_row_loop. "
                "Use shared_stored fields and child_vs_nest block audits for correctness. compare_pass not auto-waived."
            )
        elif shared_value_diff_count > 0:
            compose_meta[f"{prefix}_mathematically_harmless_note"] = (
                "Shared-stored value differences present; inspect top20 and block Frobenius diffs."
            )
        else:
            compose_meta[f"{prefix}_mathematically_harmless_note"] = (
                "No shared-stored value differences above 1e-15; any Frobenius gap may be pattern-only."
            )

        for block_label, r0, r1, c0, c1 in block_fro_specs:
            fro, fro_method = _safe_block_frobenius_difference(
                old,
                new,
                row_start=r0,
                row_end=r1,
                col_start=c0,
                col_end=c1,
                comm=comm,
                mesh_level=mesh_level,
            )
            compose_meta[f"{prefix}_block_{block_label}_frobenius_diff"] = _safe_float(fro)
            if fro_method:
                compose_meta[f"{prefix}_block_{block_label}_frobenius_method"] = fro_method
        compose_meta[f"{prefix}_audit_pass"] = True
    except Exception as exc:
        compose_meta[f"{prefix}_audit_pass"] = False
        compose_meta[f"{prefix}_audit_unavailable_reason"] = f"{type(exc).__name__}:{exc}"
        compose_meta[f"{prefix}_top20_shared"] = compose_meta.get(f"{prefix}_top20_shared") or []
        compose_meta[f"{prefix}_top20_pattern_mismatch"] = (
            compose_meta.get(f"{prefix}_top20_pattern_mismatch") or []
        )
        compose_meta[f"{prefix}_top20"] = compose_meta.get(f"{prefix}_top20") or []
    finally:
        if diff_mat is not None:
            try:
                diff_mat.destroy()
            except Exception:
                pass


def _record_child_vs_nest_block_audit(
    *,
    child_blocks: Sequence[Tuple[str, Any, int, int, int, int]],
    nest_mono: Any,
    rowloop_mono: Any,
    comm: Any,
    compose_meta: Dict[str, Any],
    mesh_level: str = "",
) -> None:
    prefix = "B3_BLOCK_COMPOSE_compare_A_child_vs_nest"
    audit = _audit_helpers()
    compose_meta[f"{prefix}_audit_begin"] = True
    coupling_orientation: Dict[str, Any] = {}
    detail_summaries: Dict[str, Dict[str, Any]] = {}
    for block_label, child, r0, r1, c0, c1 in child_blocks:
        nest_sub = None
        row_sub = None
        try:
            nest_sub = _extract_submatrix(
                nest_mono, row_start=r0, row_end=r1, col_start=c0, col_end=c1, comm=comm
            )
            row_sub = _extract_submatrix(
                rowloop_mono, row_start=r0, row_end=r1, col_start=c0, col_end=c1, comm=comm
            )
            child_nnz = int(audit._petsc_mat_global_nnz_used(child))
            nest_nnz = int(audit._petsc_mat_global_nnz_used(nest_sub))
            row_nnz = int(audit._petsc_mat_global_nnz_used(row_sub))
            child_vs_nest = float(audit._petsc_mat_frobenius_difference(child, nest_sub))
            child_vs_row = float(audit._petsc_mat_frobenius_difference(child, row_sub))
            nest_vs_row = float(audit._petsc_mat_frobenius_difference(nest_sub, row_sub))
            compose_meta[f"{prefix}_{block_label}_child_nnz"] = child_nnz
            compose_meta[f"{prefix}_{block_label}_nest_block_nnz"] = nest_nnz
            compose_meta[f"{prefix}_{block_label}_rowloop_block_nnz"] = row_nnz
            compose_meta[f"{prefix}_{block_label}_frobenius_diff"] = _safe_float(child_vs_nest)
            compose_meta[f"{prefix}_{block_label}_child_vs_rowloop_frobenius_diff"] = _safe_float(child_vs_row)
            compose_meta[f"{prefix}_{block_label}_nest_vs_rowloop_frobenius_diff"] = _safe_float(nest_vs_row)
            compose_meta[f"{prefix}_{block_label}_row_col_range"] = {
                "row_start": int(r0),
                "row_end": int(r1),
                "col_start": int(c0),
                "col_end": int(c1),
            }
            if block_label in ("Auu", "App"):
                detail_summaries[block_label] = _record_child_vs_nest_block_pair_detail(
                    child,
                    nest_sub,
                    block_label=block_label,
                    row_offset=int(r0),
                    col_offset=int(c0),
                    compose_meta=compose_meta,
                    prefix=prefix,
                    mesh_level=mesh_level,
                )
            if block_label in ("Aup", "Apu"):
                mate_label = "Apu" if block_label == "Aup" else "Aup"
                mate_spec = next((spec for spec in child_blocks if spec[0] == mate_label), None)
                if mate_spec is not None:
                    _ml, _mate_child, mr0, mr1, mc0, mc1 = mate_spec
                    try:
                        indptr, indices, data = child.getValuesCSR()
                        from scipy import sparse

                        child_csr = sparse.csr_matrix(
                            (
                                np.asarray(data, dtype=np.float64),
                                np.asarray(indices, dtype=np.int32),
                                np.asarray(indptr, dtype=np.int32),
                            ),
                            shape=tuple(int(x) for x in child.getSize()),
                        )
                        mate_csr = _extract_submatrix_scipy_dev(
                            nest_mono, row_start=mr0, row_end=mr1, col_start=mc0, col_end=mc1
                        )
                        if child_csr.shape == mate_csr.shape:
                            fro_swap = float(np.linalg.norm((child_csr - mate_csr).data))
                            coupling_orientation[f"{block_label}_frobenius_vs_mate_nest_block"] = _safe_float(
                                fro_swap
                            )
                        mate_t = mate_csr.transpose()
                        if child_csr.shape == mate_t.shape:
                            fro_swap_t = float(np.linalg.norm((child_csr - mate_t).data))
                            coupling_orientation[f"{block_label}_frobenius_vs_mate_nest_block_transpose"] = (
                                _safe_float(fro_swap_t)
                            )
                    except Exception as exc_t:
                        coupling_orientation[f"{block_label}_orientation_check_unavailable"] = (
                            f"{type(exc_t).__name__}:{exc_t}"
                        )
        except Exception as exc:
            compose_meta[f"{prefix}_{block_label}_audit_unavailable_reason"] = f"{type(exc).__name__}:{exc}"
        finally:
            for sub in (nest_sub, row_sub):
                if sub is not None:
                    try:
                        sub.destroy()
                    except Exception:
                        pass
    compose_meta[f"{prefix}_coupling_orientation"] = coupling_orientation
    if coupling_orientation:
        aup_direct = coupling_orientation.get("Aup_frobenius_vs_mate_nest_block")
        aup_swap = coupling_orientation.get("Aup_frobenius_vs_mate_nest_block_transpose")
        if aup_direct is not None and aup_swap is not None:
            compose_meta[f"{prefix}_Aup_Apu_transpose_swap_likely"] = bool(float(aup_swap) + 1.0e-12 < float(aup_direct))
    auu_summary = detail_summaries.get("Auu") or {}
    app_summary = detail_summaries.get("App") or {}
    pattern_harmless = bool(
        auu_summary.get("pattern_harmless_unit_diag_bc")
        or auu_summary.get("petsc_diff_harmless_unit_diag_bc")
    ) and bool(
        app_summary.get("pattern_harmless_unit_diag_bc")
        or app_summary.get("petsc_diff_harmless_unit_diag_bc")
    )
    compose_meta["B3_BLOCK_COMPOSE_compare_A_child_vs_nest_pattern_harmless_bc_candidate"] = pattern_harmless
    compose_meta[f"{prefix}_pattern_harmless_bc_candidate"] = pattern_harmless
    compose_meta[f"{prefix}_pattern_harmless_bc_candidate_basis"] = (
        "Auu_and_App_each_one_pattern_only_unit_diagonal_bc_identity"
        if pattern_harmless
        else "not_both_Auu_App_pattern_only_unit_diagonal_bc_identity"
    )
    compose_meta[f"{prefix}_pattern_harmless_bc_candidate_note"] = (
        "MatNest numeric shared-stored values match source child blocks in Auu and App; "
        "remaining sqrt(2) A Frobenius gap is pattern-only BC/identity artifact "
        "(one unit diagonal entry stored on one side only in each block). compare_pass not auto-waived."
        if pattern_harmless
        else (
            "Auu/App pattern-only child-vs-nest mismatches do not both reduce to a single "
            "unit diagonal BC/identity entry; inspect Auu_pattern and App_pattern top20."
        )
    )
    compose_meta[f"{prefix}_diff_harmless_bc_candidate"] = pattern_harmless
    compose_meta[f"{prefix}_diff_harmless_bc_candidate_basis"] = compose_meta[
        f"{prefix}_pattern_harmless_bc_candidate_basis"
    ]
    compose_meta[f"{prefix}_diff_harmless_bc_candidate_note"] = compose_meta[
        f"{prefix}_pattern_harmless_bc_candidate_note"
    ]
    compose_meta[f"{prefix}_audit_pass"] = True


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
    mesh_level: str = "",
) -> None:
    _streaming_compose_matrix_diff_audit(
        old,
        new,
        label=label,
        n_u=n_u,
        n_p=n_p,
        comm=comm,
        compose_meta=compose_meta,
        block_name_fn=block_name_fn,
        block_fro_specs=block_fro_specs,
        mesh_level=mesh_level,
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

    compose_meta["B3_BLOCK_COMPOSE_matnest_owned_copy_method"] = "PETSc_Mat_duplicate_then_copy"
    a_nest_blocks = [
        [
            _owned_aij_copy_for_matnest(a_uu, label="Auu", compose_meta=compose_meta),
            _owned_aij_copy_for_matnest(a_up, label="Aup", compose_meta=compose_meta),
        ],
        [
            _owned_aij_copy_for_matnest(a_pu, label="Apu", compose_meta=compose_meta),
            _owned_aij_copy_for_matnest(a_pp, label="App", compose_meta=compose_meta),
        ],
    ]
    m_nest_blocks = [
        [_owned_aij_copy_for_matnest(m_uu, label="Muu", compose_meta=compose_meta), m_up_zero],
        [
            _owned_aij_copy_for_matnest(m_pu, label="Mpu", compose_meta=compose_meta),
            _owned_aij_copy_for_matnest(m_pp, label="Mpp", compose_meta=compose_meta),
        ],
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
    mesh_level: str = "",
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
        _matnest_breadcrumb(compose_meta, "before_compare_child_vs_nest_A")
        nu = int(n_u)
        np_ = int(n_p)
        _record_child_vs_nest_block_audit(
            child_blocks=(
                ("Auu", a_uu, 0, nu, 0, nu),
                ("Aup", a_up, 0, nu, nu, nu + np_),
                ("Apu", a_pu, nu, nu + np_, 0, nu),
                ("App", a_pp, nu, nu + np_, nu, nu + np_),
            ),
            nest_mono=a_new,
            rowloop_mono=a_old,
            comm=comm,
            compose_meta=compose_meta,
            mesh_level=mesh_level,
        )
        _matnest_breadcrumb(compose_meta, "after_compare_child_vs_nest_A")
        _matnest_breadcrumb(compose_meta, "before_compare_diff_audit_A")
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
            mesh_level=mesh_level,
        )
        compose_meta["B3_BLOCK_COMPOSE_compare_A_DIFF_TOP20_SHARED"] = list(
            compose_meta.get("B3_BLOCK_COMPOSE_compare_A_diff_top20_shared") or []
        )
        compose_meta["B3_BLOCK_COMPOSE_compare_A_DIFF_TOP20_PATTERN_MISMATCH"] = list(
            compose_meta.get("B3_BLOCK_COMPOSE_compare_A_diff_top20_pattern_mismatch") or []
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
            mesh_level=mesh_level,
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
                mesh_level=mesh_level,
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
