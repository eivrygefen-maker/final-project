"""Mode displacement vectors: relative noise floor + float32 for the FEM ROM pipeline."""
from __future__ import annotations

import numpy as np

# Zero entries with |x| < eps * max(|vector|) (per mode), then cast to float32.
MODE_VECTOR_RELATIVE_EPS = 1e-7


def sparsify_relative_then_float32(vec: np.ndarray) -> np.ndarray:
    """
    Apply relative near-zero thresholding, then store as float32 (1-D column layout compatible).

    If max(|v|) is 0, returns a float32 zero vector of the same length.
    """
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    amax = float(np.max(np.abs(v)))
    if amax > 0.0:
        cutoff = MODE_VECTOR_RELATIVE_EPS * amax
        v = np.where(np.abs(v) < cutoff, 0.0, v)
    return np.asarray(v, dtype=np.float32)
