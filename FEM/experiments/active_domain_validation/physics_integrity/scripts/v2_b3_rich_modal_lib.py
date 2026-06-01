#!/usr/bin/env python3
"""Shared helpers for rich modal export v1 (active basis + W prolongation)."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from v2_b3_petsc_util import write_json_atomic


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v

RICH_MODAL_SCHEMA = "b3_rich_modal_manifest_v1"
SYNTHESIS_METADATA_SCHEMA = "b3_synthesis_metadata_v1"
RICH_MODAL_POST_SCHEMA = "b3_rich_modal_post_v1"
REGION_DOF_LAYOUT = "B3_W_global_row_indices_via_u_idx_p_idx"

SYNTHESIS_METADATA_JSON = "synthesis_metadata.json"
REGION_DOF_INDICES_NPZ = "region_dof_indices.npz"
RICH_MODAL_DIRNAME = "rich_modal"
RICH_MODAL_MANIFEST_JSON = "rich_modal_manifest.json"
MODES_ACTIVE_NPZ = "modes_active.npz"
MODES_CATALOG_JSONL = "modes_catalog.jsonl"

TAG_PROTOCOL_V1 = {
    "Top_Plate": 1,
    "Soundhole": 2,
    "Back_Plate": 3,
    "Ribs_Sides": 4,
    "wood_fix": 5,
    "Air_Internal": 10,
}


def normalization_convention_v1() -> Dict[str, Any]:
    return {
        "problem_type": "GNHEP",
        "eps_type": "KRYLOVSCHUR",
        "st_type": "SINVERT",
        "eigenvector_basis": "active_reduced_A_active",
        "eigenpair_storage": "real_part_only",
        "eigenvector_dtype": "float64",
        "st_shift_applied_to": "shift_invert_center_lambda_sq",
        "physical_frequency_hz": "sqrt(lambda_real)/(2*pi) when lambda_imag≈0",
        "damping_in_eigensolve": False,
        "damping_applied_in": "STK_or_audio_post",
        "note": (
            "Vectors are SLEPc Ritz coefficients in active DOF layout; not mass-normalized. "
            "Do not label outputs as microphone pressure; use audio_output_proxy wording only."
        ),
    }


def prolongate_active_to_W(x_active: np.ndarray, built: Dict[str, Any]) -> np.ndarray:
    """Map active-vector coefficients to full B3 W row layout (length n_w)."""
    active_local = np.asarray(built["active_local"], dtype=np.int32).ravel()
    free_rows = np.asarray(built["free_rows"], dtype=np.int32).ravel()
    n_w = int(built["n_w"])
    n_free = int(free_rows.size)
    x_active = np.asarray(x_active, dtype=np.float64).ravel()
    if int(x_active.size) != int(active_local.size):
        raise ValueError(f"active vector length {x_active.size} != active_local {active_local.size}")
    x_free = np.zeros(n_free, dtype=np.float64)
    x_free[active_local] = x_active
    x_full = np.zeros(n_w, dtype=np.float64)
    x_full[free_rows] = x_free
    return x_full


def participation_energy_fraction(x_full: np.ndarray, indices: np.ndarray) -> float:
    """‖x[indices]‖² / ‖x‖² with safe denominator."""
    idx = np.asarray(indices, dtype=np.int32).ravel()
    if idx.size == 0:
        return 0.0
    x = np.asarray(x_full, dtype=np.float64).ravel()
    total = float(np.dot(x, x))
    if total <= 0.0:
        return 0.0
    part = x[idx]
    return float(np.dot(part, part) / total)


def scalar_block_norms(x_full: np.ndarray, built: Dict[str, Any]) -> Dict[str, float]:
    u_idx = np.asarray(built["u_idx"], dtype=np.int32).ravel()
    p_idx = np.asarray(built["p_idx"], dtype=np.int32).ravel()
    u_norm = float(np.linalg.norm(np.abs(x_full[u_idx])))
    p_norm = float(np.linalg.norm(np.abs(x_full[p_idx])))
    x_norm = float(np.linalg.norm(x_full))
    p_support = p_norm / max(x_norm, 1.0e-30)
    return {
        "u_norm_W": u_norm,
        "p_norm_W": p_norm,
        "x_norm_W": x_norm,
        "p_support": p_support,
    }


class RichModalCollector:
    """Accumulate accepted modes across ST targets for one solve run."""

    def __init__(self) -> None:
        self._columns: List[np.ndarray] = []
        self._catalog: List[Dict[str, Any]] = []

    def add_mode(
        self,
        *,
        x_active: np.ndarray,
        target_index: int,
        target_hz: float,
        record: Dict[str, Any],
    ) -> None:
        self._columns.append(np.asarray(x_active, dtype=np.float64).ravel().copy())
        entry = dict(record)
        entry["target_index"] = int(target_index)
        entry["st_shift_target_hz"] = float(target_hz)
        entry["catalog_index"] = len(self._catalog)
        self._catalog.append(entry)

    @property
    def mode_count(self) -> int:
        return len(self._columns)

    def write_bundle(
        self,
        rich_dir: Path,
        *,
        checkpoint_dir: Path,
        solve_output_dir: Path,
        factor_solver: str,
        nev: int,
        ncv: int,
        target_set: str,
        targets_hz: Sequence[float],
        acceptance_interval_hz: Sequence[float],
        synthesis_metadata_path: Path,
    ) -> Dict[str, Any]:
        rich_dir.mkdir(parents=True, exist_ok=True)
        if not self._columns:
            manifest = {
                "schema": RICH_MODAL_SCHEMA,
                "mode_count": 0,
                "status": "empty",
            }
            write_json_atomic(rich_dir / RICH_MODAL_MANIFEST_JSON, manifest)
            return manifest

        n_active = int(self._columns[0].size)
        mat = np.column_stack(self._columns)
        cat = self._catalog
        np.savez_compressed(
            rich_dir / MODES_ACTIVE_NPZ,
            eigenvectors_active=mat,
            active_dimension=np.int64(n_active),
            mode_index=np.asarray([c.get("eps_slot_index", c.get("mode_index", -1)) for c in cat], dtype=np.int32),
            frequency_hz=np.asarray([c["frequency_hz"] for c in cat], dtype=np.float64),
            lambda_real=np.asarray([c.get("lambda_real", np.nan) for c in cat], dtype=np.float64),
            lambda_imag=np.asarray([c.get("lambda_imag", np.nan) for c in cat], dtype=np.float64),
            st_shift_target_hz=np.asarray([c["st_shift_target_hz"] for c in cat], dtype=np.float64),
            target_index=np.asarray([c["target_index"] for c in cat], dtype=np.int32),
            eps_slot_index=np.asarray([c.get("eps_slot_index", c.get("mode_index", -1)) for c in cat], dtype=np.int32),
            eps_compute_error_relative=np.asarray(
                [c.get("eps_compute_error_relative", np.nan) for c in cat], dtype=np.float64
            ),
            u_norm_W=np.asarray([c.get("u_norm_W", np.nan) for c in cat], dtype=np.float64),
            p_norm_W=np.asarray([c.get("p_norm_W", np.nan) for c in cat], dtype=np.float64),
            p_support=np.asarray([c.get("p_support", np.nan) for c in cat], dtype=np.float64),
            x_norm_W=np.asarray([c.get("x_norm_W", np.nan) for c in cat], dtype=np.float64),
        )
        catalog_path = rich_dir / MODES_CATALOG_JSONL
        with catalog_path.open("w", encoding="utf-8") as fh:
            for row in cat:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

        freqs = [float(c["frequency_hz"]) for c in cat]
        manifest: Dict[str, Any] = {
            "schema": RICH_MODAL_SCHEMA,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "checkpoint_dir": str(checkpoint_dir.resolve()),
            "solve_output_dir": str(solve_output_dir.resolve()),
            "factor_solver": factor_solver,
            "nev": int(nev),
            "ncv": int(ncv),
            "target_set": str(target_set),
            "targets_hz": list(targets_hz),
            "acceptance_interval_hz": list(acceptance_interval_hz),
            "normalization_convention": normalization_convention_v1(),
            "synthesis_metadata_ref": str(synthesis_metadata_path.resolve()),
            "modes_active_npz": str((rich_dir / MODES_ACTIVE_NPZ).resolve()),
            "modes_catalog_jsonl": str(catalog_path.resolve()),
            "mode_count": len(cat),
            "unique_frequency_count": len({round(f, 6) for f in freqs}),
            "duplicate_frequency_count": len(freqs) - len({round(f, 6) for f in freqs}),
            "deduplication_note": (
                "Duplicates across shifts are retained in v1; use catalog for MAC/frequency dedupe."
            ),
        }
        write_json_atomic(rich_dir / RICH_MODAL_MANIFEST_JSON, manifest)
        return manifest


def frequency_dedupe_report(
    modes: Sequence[Dict[str, Any]],
    *,
    tol_hz: float,
) -> Dict[str, Any]:
    """Report duplicate frequencies without removing modes silently."""
    groups: Dict[float, List[int]] = {}
    for i, row in enumerate(modes):
        f = float(row["frequency_hz"])
        key = round(f / tol_hz) * tol_hz if tol_hz > 0 else round(f, 6)
        groups.setdefault(float(key), []).append(int(i))
    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
    return {
        "tolerance_hz": float(tol_hz),
        "unique_frequency_groups": len(groups),
        "duplicate_groups": len(dup_groups),
        "duplicate_group_keys_hz": [safe_float(k) for k in sorted(dup_groups.keys())],
        "duplicate_indices_by_group": dup_groups,
    }
