#!/usr/bin/env python3
"""Checkpoint built_metadata normalization (no PETSc / DOLFINx)."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

CHECKPOINT_METADATA_REQUIRED_KEYS = (
    "mesh_level",
    "active_dimension",
    "active_local",
    "inactive_local",
    "free_rows",
    "bc_rows",
    "u_idx",
    "p_idx",
    "n_w",
    "n_u_b3",
)


def normalize_checkpoint_metadata(meta: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], bool]:
    """Validate and fill optional inactive-count fields in built_metadata.json."""
    missing_required = [k for k in CHECKPOINT_METADATA_REQUIRED_KEYS if k not in meta]
    if missing_required:
        return dict(meta), missing_required, False
    normalized: Dict[str, Any] = dict(meta)
    inactive_local = np.asarray(meta["inactive_local"], dtype=np.int32).ravel()
    inactive_n = int(inactive_local.size)
    for key, default in (
        ("inactive_structural_count", inactive_n),
        ("inactive_pressure_count", 0),
        ("inactive_aup_overlap_count", 0),
        ("aup_supported_count", 0),
        ("parent_raw_Auu_exact_zero_count", inactive_n),
        ("parent_raw_Auu_nonzero_count", 0),
    ):
        normalized.setdefault(key, default)
    active_dim = int(normalized.get("active_dimension", 0))
    active_local = np.asarray(normalized["active_local"], dtype=np.int32).ravel()
    schema_pass = bool(active_dim > 0 and int(active_local.size) == active_dim and inactive_n >= 0)
    return normalized, missing_required, schema_pass
