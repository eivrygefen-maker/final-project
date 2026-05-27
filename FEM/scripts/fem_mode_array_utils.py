"""Mode displacement vectors: relative sparsification + float32 CSR for the FEM ROM pipeline."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Union

import numpy as np
from scipy import sparse

# Zero entries with |x| < eps * max(|vector|) (per mode), then store as float32 CSR column (N, 1).
MODE_VECTOR_RELATIVE_EPS = 1e-7

# Worker / merge artifact: one CSR matrix per file (scipy.sparse.save_npz format).
MODE_VECTOR_FILE_SUFFIX = ".smx.npz"
# Diagnostic-only lossless dense column (float64); not used by production merge paths.
MODE_VECTOR_DENSE_LOSSLESS_SUFFIX = ".smx.dense.npy"


def sparsify_relative_then_float32(vec: np.ndarray) -> np.ndarray:
    """
    Apply relative near-zero thresholding, then return a dense float32 column vector (N,).

    If max(|v|) is 0, returns a float32 zero vector of the same length.
    """
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    amax = float(np.max(np.abs(v)))
    if amax > 0.0:
        cutoff = MODE_VECTOR_RELATIVE_EPS * amax
        v = np.where(np.abs(v) < cutoff, 0.0, v)
    return np.asarray(v, dtype=np.float32)


def dense_to_csr_f32_column(vec: np.ndarray) -> sparse.csr_matrix:
    """Threshold relative to peak amplitude, then CSR (N, 1) float32."""
    col = sparsify_relative_then_float32(vec).reshape(-1, 1)
    m = sparse.csr_matrix(col)
    m.eliminate_zeros()
    return m.astype(np.float32, copy=False)


def load_mode_column_any(path: Union[str, Path]) -> sparse.csr_matrix:
    """Load a mode column as CSR float32 from ``*.smx.npz`` (scipy) or legacy ``.npy`` dense."""
    p = Path(path)
    if p.suffix.lower() == ".npy":
        return dense_to_csr_f32_column(np.load(str(p)))
    if p.name.endswith(MODE_VECTOR_FILE_SUFFIX):
        m = sparse.load_npz(str(p))
        if not sparse.issparse(m):
            raise TypeError(f"Expected sparse matrix in {p}")
        return m.tocsr().astype(np.float32, copy=False)
    raise ValueError(f"Unsupported mode vector file (use .npy or *{MODE_VECTOR_FILE_SUFFIX}): {p}")


def csr_u_slice(mat: sparse.csr_matrix, n_u: int) -> sparse.csr_matrix:
    """First n_u rows (structural displacement block), CSR."""
    return mat[: int(n_u), :].tocsr(copy=False)


def csr_col_norm(mat: sparse.csr_matrix) -> float:
    """Euclidean norm of a sparse column vector."""
    s = float(mat.multiply(mat).sum())
    if s <= 0.0:
        return 0.0
    return math.sqrt(s)


def csr_col_dot(a: sparse.csr_matrix, b: sparse.csr_matrix) -> float:
    """Dot product of two same-height sparse column vectors (Hadamard sum)."""
    return float(a.multiply(b).sum())


def csr_normalized_overlap(va: sparse.csr_matrix, vb: sparse.csr_matrix) -> float:
    """|va·vb| / (||va|| ||vb||), clipped to [0, 1]."""
    na = csr_col_norm(va)
    nb = csr_col_norm(vb)
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    ov = abs(csr_col_dot(va, vb)) / max(na * nb, 1e-30)
    return float(np.clip(ov, 0.0, 1.0))


def save_mode_csr(path: Union[str, Path], mat: sparse.csr_matrix) -> None:
    """Write CSR float32 with scipy (single-file compressed sparse NPZ)."""
    m = mat.tocsr().astype(np.float32, copy=False)
    m.eliminate_zeros()
    sparse.save_npz(str(path), m, compressed=True)


def save_mode_dense_f64_lossless(path: Union[str, Path], vec: np.ndarray) -> None:
    """Diagnostic-only: lossless float64 dense column for replay (no relative sparsification)."""
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    np.save(str(path), arr)


def load_mode_dense_f64_lossless(path: Union[str, Path]) -> np.ndarray:
    """Load diagnostic lossless dense vector."""
    return np.asarray(np.load(str(path)), dtype=np.float64).ravel()


def is_lossless_dense_mode_path(path: Union[str, Path]) -> bool:
    return str(path).endswith(MODE_VECTOR_DENSE_LOSSLESS_SUFFIX)
