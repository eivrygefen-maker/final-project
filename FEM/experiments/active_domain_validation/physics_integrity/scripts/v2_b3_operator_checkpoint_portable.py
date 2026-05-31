#!/usr/bin/env python3
"""Portable CSR checkpoint export/load for cross-PETSc-version solver benchmarks."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from petsc4py import PETSc

import run_v2_B3_trace_coupled_operator_and_seed_transfer_audit as audit

A_CSR_NPZ = "A_active_csr.npz"
M_CSR_NPZ = "M_active_csr.npz"
CSR_METADATA_JSON = "csr_metadata.json"

B3_ST_CHECKPOINT_PORTABLE_SMOKE_ARG = "--B3-ST-checkpoint-portable-smoke-only"
B3_ST_REUSE_CHECKPOINT_ARG = "--B3-ST-reuse-checkpoint-dir"

_MATRIX_KEYS = ("A_active", "M_active")
_CSR_NPZ_BY_KEY = {"A_active": A_CSR_NPZ, "M_active": M_CSR_NPZ}


def _parse_arg_value(argv: Sequence[str], flag: str) -> Optional[str]:
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return str(argv[i + 1])
        if arg.startswith(f"{flag}="):
            return str(arg.split("=", 1)[1])
    return None


def is_checkpoint_portable_smoke_mode(argv: Sequence[str]) -> bool:
    return B3_ST_CHECKPOINT_PORTABLE_SMOKE_ARG in argv


def _petsc_version_string() -> str:
    try:
        return str(PETSc.Sys.getVersion())
    except Exception:
        return "unknown"


def _scalar_type_label() -> str:
    try:
        return str(PETSc.ScalarType)
    except Exception:
        return "unknown"


def _precision_label() -> str:
    try:
        return str(PETSc.RealType)
    except Exception:
        return "unknown"


def _indices_sorted(indptr: np.ndarray, indices: np.ndarray) -> bool:
    if indices.size == 0:
        return True
    for row in range(int(indptr.size) - 1):
        start = int(indptr[row])
        end = int(indptr[row + 1])
        if end <= start:
            continue
        cols = indices[start:end]
        if cols.size <= 1:
            continue
        if not bool(np.all(cols[1:] >= cols[:-1])):
            return False
    return True


def _mat_frobenius_norm(mat: Any) -> Optional[float]:
    try:
        return float(mat.norm(PETSc.Norm.FROBENIUS))
    except Exception:
        return None


def _extract_csr_arrays(mat: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int], int]:
    audit._petsc_mat_try_assemble(mat)
    indptr, indices, data = mat.getValuesCSR()
    indptr = np.asarray(indptr)
    indices = np.asarray(indices)
    data = np.asarray(data)
    n_rows, n_cols = audit._mat_shape(mat)
    nnz = int(data.size)
    return indptr, indices, data, (int(n_rows), int(n_cols)), nnz


def _matrix_csr_entry(
    mat: Any,
    *,
    label: str,
    filename: str,
) -> Dict[str, Any]:
    indptr, indices, data, shape, nnz = _extract_csr_arrays(mat)
    return {
        "label": label,
        "filename": filename,
        "shape": list(shape),
        "nnz": nnz,
        "data_dtype": str(data.dtype),
        "index_dtype": str(indices.dtype),
        "indptr_dtype": str(indptr.dtype),
        "indices_sorted": bool(_indices_sorted(indptr, indices)),
        "frobenius_norm": _mat_frobenius_norm(mat),
    }


def export_portable_csr_checkpoint(
    checkpoint: Path,
    *,
    A_active: Any,
    M_active: Any,
) -> Dict[str, Any]:
    checkpoint.mkdir(parents=True, exist_ok=True)
    mats = {"A_active": A_active, "M_active": M_active}
    matrix_meta: Dict[str, Any] = {}
    exported_files: List[str] = []

    for key, mat in mats.items():
        indptr, indices, data, shape, nnz = _extract_csr_arrays(mat)
        npz_name = _CSR_NPZ_BY_KEY[key]
        npz_path = checkpoint / npz_name
        np.savez_compressed(
            npz_path,
            indptr=indptr,
            indices=indices,
            data=data,
            shape=np.asarray(shape, dtype=np.int64),
        )
        exported_files.append(npz_name)
        matrix_meta[key] = _matrix_csr_entry(mat, label=key, filename=npz_name)

    body: Dict[str, Any] = {
        "export_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "format": "csr_npz_v1",
        "petsc_version_export_env": _petsc_version_string(),
        "scalar_type": _scalar_type_label(),
        "precision": _precision_label(),
        "matrices": matrix_meta,
    }
    meta_path = checkpoint / CSR_METADATA_JSON
    meta_path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    exported_files.append(CSR_METADATA_JSON)
    return {
        "csr_metadata_json": str(meta_path.resolve()),
        "csr_exported_files": exported_files,
        **body,
    }


def _load_csr_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    with np.load(path, allow_pickle=False) as z:
        indptr = np.asarray(z["indptr"])
        indices = np.asarray(z["indices"])
        data = np.asarray(z["data"])
        if "shape" in z:
            shape_arr = np.asarray(z["shape"]).ravel()
            shape = (int(shape_arr[0]), int(shape_arr[1]))
        else:
            shape = (int(indptr.size - 1), int(indptr.size - 1))
    return indptr, indices, data, shape


def _petsc_aij_from_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    *,
    shape: Tuple[int, int],
    comm: Any,
) -> Any:
    n_rows, n_cols = shape
    mat = None
    last_exc: Optional[Exception] = None
    for factory in (
        lambda: PETSc.Mat().createAIJ(size=(n_rows, n_cols), csr=(indptr, indices, data), comm=comm),
        lambda: PETSc.Mat().createAIJ(
            size=(n_rows, n_cols),
            csr=((indptr, indices, data), (indptr, indices, data)),
            comm=comm,
        ),
    ):
        try:
            mat = factory()
            break
        except Exception as exc:
            last_exc = exc
    if mat is None:
        mat = PETSc.Mat().create(comm=comm)
        mat.setSizes([n_rows, n_cols])
        mat.setType("aij")
        mat.setUp()
        try:
            mat.setValuesCSR(indptr, indices, data)
        except Exception as exc_set:
            raise RuntimeError(
                f"CSR rebuild failed: setValuesCSR={exc_set}; createAIJ={last_exc}"
            ) from exc_set
    audit._petsc_mat_try_assemble(mat)
    return mat


def _load_mat_binary(path: Path) -> Any:
    viewer = PETSc.Viewer().createBinary(str(path), "r", comm=PETSc.COMM_WORLD)
    try:
        mat = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
        mat.load(viewer)
        audit._petsc_mat_try_assemble(mat)
        return mat
    finally:
        viewer.destroy()


def _load_mat_csr(checkpoint: Path, *, key: str) -> Any:
    csr_meta_path = checkpoint / CSR_METADATA_JSON
    if not csr_meta_path.is_file():
        raise FileNotFoundError(f"missing {CSR_METADATA_JSON} under {checkpoint}")
    csr_meta = json.loads(csr_meta_path.read_text(encoding="utf-8"))
    entry = (csr_meta.get("matrices") or {}).get(key)
    if not entry:
        raise KeyError(f"csr metadata missing matrix key {key!r}")
    npz_path = checkpoint / str(entry.get("filename") or _CSR_NPZ_BY_KEY[key])
    if not npz_path.is_file():
        raise FileNotFoundError(f"missing CSR npz for {key}: {npz_path}")
    indptr, indices, data, shape = _load_csr_npz(npz_path)
    return _petsc_aij_from_csr(indptr, indices, data, shape=shape, comm=PETSc.COMM_WORLD)


def _verify_loaded_mat_against_csr_metadata(
    mat: Any,
    *,
    key: str,
    csr_meta: Dict[str, Any],
    tol: float = 1.0e-6,
) -> Dict[str, Any]:
    entry = (csr_meta.get("matrices") or {}).get(key) or {}
    shape = audit._mat_shape(mat)
    indptr, indices, data, loaded_shape, nnz = _extract_csr_arrays(mat)
    expected_shape = tuple(int(x) for x in (entry.get("shape") or loaded_shape))
    expected_nnz = int(entry.get("nnz") or 0)
    expected_norm = entry.get("frobenius_norm")
    loaded_norm = _mat_frobenius_norm(mat)
    norm_ok = True
    norm_delta = None
    if expected_norm is not None and loaded_norm is not None:
        try:
            norm_delta = abs(float(loaded_norm) - float(expected_norm))
            norm_ok = bool(norm_delta <= max(tol, tol * abs(float(expected_norm))))
        except (TypeError, ValueError):
            norm_ok = False
    return {
        "matrix_key": key,
        "shape_expected": list(expected_shape),
        "shape_loaded": list(shape),
        "shape_match": bool(tuple(shape) == tuple(expected_shape)),
        "nnz_expected": expected_nnz,
        "nnz_loaded": int(nnz),
        "nnz_match": bool(int(nnz) == expected_nnz),
        "indices_sorted_loaded": bool(_indices_sorted(indptr, indices)),
        "indices_sorted_expected": bool(entry.get("indices_sorted")),
        "frobenius_norm_expected": expected_norm,
        "frobenius_norm_loaded": loaded_norm,
        "frobenius_norm_delta": norm_delta,
        "frobenius_norm_match": bool(norm_ok),
        "verification_pass": bool(
            tuple(shape) == tuple(expected_shape)
            and int(nnz) == expected_nnz
            and norm_ok
        ),
    }


def load_operators_with_portable_fallback(
    checkpoint: Path,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load A/M from PETSc binary; on failure rebuild from portable CSR arrays."""
    diag: Dict[str, Any] = {
        "checkpoint_dir": str(checkpoint.resolve()),
        "load_path_by_matrix": {},
        "binary_load_errors": {},
    }
    mats: Dict[str, Any] = {}
    binary_names = {"A_active": "A_active.petsc.bin", "M_active": "M_active.petsc.bin"}

    for key, bin_name in binary_names.items():
        bin_path = checkpoint / bin_name
        if bin_path.is_file():
            try:
                mats[key] = _load_mat_binary(bin_path)
                diag["load_path_by_matrix"][key] = "petsc_binary"
                continue
            except Exception as exc:
                diag["binary_load_errors"][key] = f"{type(exc).__name__}:{exc}"
        try:
            mats[key] = _load_mat_csr(checkpoint, key=key)
            diag["load_path_by_matrix"][key] = "csr_rebuild"
        except Exception as exc:
            diag["load_path_by_matrix"][key] = "failed"
            diag["csr_load_error"] = diag.get("csr_load_error") or {}
            diag["csr_load_error"][key] = f"{type(exc).__name__}:{exc}"
            raise RuntimeError(
                f"checkpoint_load_failed:{key}: binary and CSR fallback both failed"
            ) from exc

    csr_meta_path = checkpoint / CSR_METADATA_JSON
    if csr_meta_path.is_file():
        csr_meta = json.loads(csr_meta_path.read_text(encoding="utf-8"))
        diag["csr_metadata_present"] = True
        diag["petsc_version_export_env"] = csr_meta.get("petsc_version_export_env")
        diag["csr_verification"] = {
            key: _verify_loaded_mat_against_csr_metadata(mats[key], key=key, csr_meta=csr_meta)
            for key in _MATRIX_KEYS
        }
        diag["csr_verification_pass"] = all(
            bool(v.get("verification_pass")) for v in diag["csr_verification"].values()
        )
    else:
        diag["csr_metadata_present"] = False
        diag["csr_verification_pass"] = None

    diag["petsc_version_load_env"] = _petsc_version_string()
    diag["load_path_summary"] = (
        "petsc_binary"
        if all(v == "petsc_binary" for v in diag["load_path_by_matrix"].values())
        else "csr_rebuild"
        if all(v == "csr_rebuild" for v in diag["load_path_by_matrix"].values())
        else "mixed"
    )
    return mats["A_active"], mats["M_active"], diag


def verify_portable_checkpoint_export(checkpoint: Path) -> Tuple[bool, List[str], Dict[str, Any]]:
    required = [
        checkpoint / "A_active.petsc.bin",
        checkpoint / "M_active.petsc.bin",
        checkpoint / "built_metadata.json",
        checkpoint / A_CSR_NPZ,
        checkpoint / M_CSR_NPZ,
        checkpoint / CSR_METADATA_JSON,
    ]
    missing = [p.name for p in required if not p.is_file()]
    detail = {
        "checkpoint_dir": str(checkpoint.resolve()),
        "required_files": [p.name for p in required],
        "missing_files": missing,
        "petsc_info_sidecars_present": all(
            (checkpoint / f"{name}.info").is_file()
            for name in ("A_active.petsc.bin", "M_active.petsc.bin")
        ),
    }
    return len(missing) == 0, missing, detail


def _probe_mkl_pardiso_lu() -> Dict[str, Any]:
    from v2_b3_st_solver_benchmark import _probe_pc_lu_factor_solver

    return _probe_pc_lu_factor_solver("mkl_pardiso")


def run_checkpoint_portable_smoke(argv: Sequence[str]) -> int:
    ckpt_raw = _parse_arg_value(argv, B3_ST_REUSE_CHECKPOINT_ARG)
    if not ckpt_raw:
        print(
            f"[B3_checkpoint_portable_smoke] requires {B3_ST_REUSE_CHECKPOINT_ARG} <dir>",
            flush=True,
        )
        return 2
    checkpoint = Path(ckpt_raw).expanduser().resolve()
    if not checkpoint.is_dir():
        print(f"[B3_checkpoint_portable_smoke] not a directory: {checkpoint}", flush=True)
        return 2

    out_path = checkpoint / "portable_checkpoint_smoke.json"
    result: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint_dir": str(checkpoint),
        "petsc_version_load_env": _petsc_version_string(),
    }
    mats: List[Any] = []
    try:
        A_active, M_active, load_diag = load_operators_with_portable_fallback(checkpoint)
        mats.extend([A_active, M_active])
        result["load"] = load_diag
        if load_diag.get("csr_verification_pass") is False:
            result["status"] = "FAIL_CSR_VERIFICATION"
            audit._write_json_atomic(out_path, result)
            print(f"[B3_checkpoint_portable_smoke] FAIL csr verification -> {out_path}", flush=True)
            return 2

        probe = _probe_mkl_pardiso_lu()
        result["mkl_pardiso_probe"] = probe
        result["status"] = "PASS" if bool(probe.get("available")) else "PASS_LOAD_PROBE_UNAVAILABLE"
        audit._write_json_atomic(out_path, result)
        print(f"[B3_checkpoint_portable_smoke] {result['status']} -> {out_path}", flush=True)
        print(
            f"[B3_checkpoint_portable_smoke] load_path={load_diag.get('load_path_summary')} "
            f"mkl_pardiso_available={probe.get('available')}",
            flush=True,
        )
        return 0 if result["status"] in ("PASS", "PASS_LOAD_PROBE_UNAVAILABLE") else 2
    except Exception as exc:
        result["status"] = "FAIL"
        result["error"] = f"{type(exc).__name__}:{exc}"
        audit._write_json_atomic(out_path, result)
        print(f"[B3_checkpoint_portable_smoke] FAIL {exc} -> {out_path}", flush=True)
        return 2
    finally:
        for mat in mats:
            try:
                mat.destroy()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(run_checkpoint_portable_smoke(sys.argv))
