#!/usr/bin/env python3
"""Structural MAC/subspace comparison for expanded harvest extension artifacts."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from v2_sensitivity_common import (
    ENERGY_ACOUSTIC_THRESHOLD,
    STRUCTURAL_MAC_CONFIDENCE_THRESHOLD,
    displacement_subspace_mac,
    harvest_ext_case_dir,
    harvest_ext_result_json,
    is_acoustic_branch,
    load_v2_mode_vector_dense,
)

SUBSPACE_MIN_COSINE_PASS = 0.75
SUBSPACE_MEAN_COSINE_PASS = 0.85
LARGE_FREQUENCY_SHIFT_HZ = 15.0


def is_structural_mode(meta: Dict[str, Any]) -> bool:
    if is_acoustic_branch(meta):
        return False
    if str(meta.get("mode_class_physical_energy")) == "structural_dominated":
        return True
    return float(meta.get("p_frac_energy_phys", 1.0)) <= 0.15


def modes_in_harvest_band(
    solve: Dict[str, Any],
    band_lo: float,
    band_hi: float,
    *,
    structural_only: bool = False,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for m in solve.get("in_band_modes") or []:
        f_hz = float(m.get("frequency_hz", float("nan")))
        if not math.isfinite(f_hz) or f_hz < band_lo or f_hz > band_hi:
            continue
        if structural_only and not is_structural_mode(m):
            continue
        rows.append(m)
    rows.sort(key=lambda r: float(r["frequency_hz"]))
    return rows


def case_spectrum_report(
    sample: Dict[str, Any],
    solve: Dict[str, Any],
    *,
    band_lo: float,
    band_hi: float,
) -> Dict[str, Any]:
    mats = sample.get("materials") or {}
    in_band = solve.get("in_band_modes") or []
    n_struct = n_acous = n_mixed = 0
    struct_freqs: List[float] = []
    for m in in_band:
        f_hz = float(m.get("frequency_hz", float("nan")))
        if not math.isfinite(f_hz) or f_hz < band_lo or f_hz > band_hi:
            continue
        cls = str(m.get("mode_class_physical_energy", ""))
        if cls == "structural_dominated" or (
            cls != "acoustic_dominated" and float(m.get("p_frac_energy_phys", 1.0)) <= 0.15
        ):
            n_struct += 1
            struct_freqs.append(f_hz)
        elif cls == "acoustic_dominated" or float(m.get("p_frac_energy_phys", 0.0)) >= ENERGY_ACOUSTIC_THRESHOLD:
            n_acous += 1
        else:
            n_mixed += 1
    return {
        "sample_id": str(sample.get("id", "")),
        "material_assignment": {
            "top_wood_id": mats.get("top_wood_id"),
            "back_wood_id": mats.get("back_wood_id"),
        },
        "harvest_band_hz": [float(band_lo), float(band_hi)],
        "number_of_converged_modes": int(solve.get("num_modes_saved", -1)),
        "nconv_marked": int((solve.get("eps_batch_diagnostics") or {}).get("nconv_marked", -1)),
        "v2_converged": bool(solve.get("v2_converged")),
        "number_of_structural_dominated_modes": n_struct,
        "number_of_acoustic_or_mixed_modes": n_acous + n_mixed,
        "number_of_acoustic_dominated_modes": n_acous,
        "number_of_mixed_modes": n_mixed,
        "structural_mode_frequency_range_hz": (
            [float(min(struct_freqs)), float(max(struct_freqs))] if struct_freqs else None
        ),
    }


def _load_mode_vector(case_dir: Path, meta: Dict[str, Any], n_W: int) -> np.ndarray:
    rel = str(meta.get("vector_path", ""))
    path = case_dir / rel if rel else Path(str(meta.get("vector_absolute_path", "")))
    if not path.is_file():
        raise FileNotFoundError(f"missing mode vector: {path}")
    return load_v2_mode_vector_dense(path, n_W)


def _mac_matrix(
    baseline_vecs: List[np.ndarray],
    material_vecs: List[np.ndarray],
    u_to_W: np.ndarray,
) -> np.ndarray:
    nb = len(baseline_vecs)
    nm = len(material_vecs)
    mac_mat = np.zeros((nb, nm), dtype=np.float64)
    for i, vb in enumerate(baseline_vecs):
        for j, vm in enumerate(material_vecs):
            mac_mat[i, j] = displacement_subspace_mac(vb, vm, u_to_W)
    return mac_mat


def hungarian_max_mac_assignment(mac_mat: np.ndarray) -> List[Tuple[int, int, float]]:
    from scipy.optimize import linear_sum_assignment

    if mac_mat.size == 0:
        return []
    cost = -np.asarray(mac_mat, dtype=np.float64)
    row_ind, col_ind = linear_sum_assignment(cost)
    return [(int(i), int(j), float(mac_mat[i, j])) for i, j in zip(row_ind, col_ind)]


def _u_block(vec: np.ndarray, u_to_W: np.ndarray) -> np.ndarray:
    u_idx = np.asarray(u_to_W, dtype=np.int32).ravel()
    return np.asarray(vec, dtype=np.float64).ravel()[u_idx]


def _orthonormal_basis(columns: List[np.ndarray]) -> np.ndarray:
    if not columns:
        return np.zeros((0, 0), dtype=np.float64)
    q, _r = np.linalg.qr(np.column_stack(columns), mode="reduced")
    return q


def subspace_overlap_metrics(basis_b: np.ndarray, basis_m: np.ndarray) -> Dict[str, Any]:
    if basis_b.size == 0 or basis_m.size == 0:
        return {
            "n_baseline_dim": int(basis_b.shape[1]) if basis_b.ndim == 2 else 0,
            "n_material_dim": int(basis_m.shape[1]) if basis_m.ndim == 2 else 0,
            "principal_angle_cosines": [],
            "subspace_overlap_mean": float("nan"),
            "subspace_overlap_min": float("nan"),
        }
    s = np.linalg.svd(basis_b.T @ basis_m, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return {
        "n_baseline_dim": int(basis_b.shape[1]),
        "n_material_dim": int(basis_m.shape[1]),
        "principal_angle_cosines": [float(x) for x in s],
        "subspace_overlap_mean": float(np.mean(s)),
        "subspace_overlap_min": float(np.min(s)),
    }


def load_structural_bank(
    sample_id: str,
    *,
    band_lo: float,
    band_hi: float,
    n_W: int,
) -> Tuple[List[Dict[str, Any]], List[np.ndarray], Dict[str, Any]]:
    result_path = harvest_ext_result_json(sample_id)
    if not result_path:
        return [], [], {"status": "missing_artifacts", "sample_id": sample_id}
    solve = json.loads(result_path.read_text(encoding="utf-8"))
    case_dir = harvest_ext_case_dir(sample_id)
    metas = modes_in_harvest_band(solve, band_lo, band_hi, structural_only=True)
    vecs: List[np.ndarray] = []
    errors: List[str] = []
    for m in metas:
        try:
            vecs.append(_load_mode_vector(case_dir, m, n_W))
        except Exception as exc:
            errors.append(f"{m.get('frequency_hz')}: {exc}")
    return metas, vecs, {"solve": solve, "load_errors": errors}


def compare_baseline_to_material(
    *,
    baseline_id: str,
    material_id: str,
    sample: Dict[str, Any],
    u_to_W: np.ndarray,
    n_W: int,
    band_lo: float,
    band_hi: float,
    criterion: Dict[str, Any],
) -> Dict[str, Any]:
    b_metas, b_vecs, b_info = load_structural_bank(
        baseline_id, band_lo=band_lo, band_hi=band_hi, n_W=n_W
    )
    m_metas, m_vecs, m_info = load_structural_bank(
        material_id, band_lo=band_lo, band_hi=band_hi, n_W=n_W
    )
    if not b_vecs or not m_vecs:
        return {
            "sample_id": material_id,
            "status": "insufficient_structural_bank",
            "n_baseline_structural": len(b_vecs),
            "n_material_structural": len(m_vecs),
        }

    mac_mat = _mac_matrix(b_vecs, m_vecs, u_to_W)
    assignment = hungarian_max_mac_assignment(mac_mat)
    assigned_rows: List[Dict[str, Any]] = []
    for i, j, mac in assignment:
        f_b = float(b_metas[i]["frequency_hz"])
        f_m = float(m_metas[j]["frequency_hz"])
        delta_f = f_m - f_b
        assigned_rows.append(
            {
                "baseline_index": i,
                "material_index": j,
                "f_baseline_hz": f_b,
                "f_material_hz": f_m,
                "delta_f_hz": delta_f,
                "structural_MAC": mac,
                "high_confidence_match": mac >= STRUCTURAL_MAC_CONFIDENCE_THRESHOLD,
                "shape_preserved_large_frequency_shift": (
                    mac >= STRUCTURAL_MAC_CONFIDENCE_THRESHOLD
                    and abs(delta_f) >= LARGE_FREQUENCY_SHIFT_HZ
                ),
            }
        )

    u_blocks_b = [_u_block(v, u_to_W) for v in b_vecs]
    u_blocks_m = [_u_block(v, u_to_W) for v in m_vecs]
    subspace = subspace_overlap_metrics(
        _orthonormal_basis(u_blocks_b), _orthonormal_basis(u_blocks_m)
    )
    subspace_pass = math.isfinite(float(subspace.get("subspace_overlap_min", float("nan")))) and (
        float(subspace["subspace_overlap_min"]) >= SUBSPACE_MIN_COSINE_PASS
        or float(subspace["subspace_overlap_mean"]) >= SUBSPACE_MEAN_COSINE_PASS
    )

    min_struct = int(criterion.get("adequate_spectrum_coverage_min_structural_modes", 8))
    min_hi = int(criterion.get("min_high_confidence_assigned_pairs", 2))
    n_hi_assigned = sum(1 for r in assigned_rows if r["high_confidence_match"])
    coverage_ok = len(b_vecs) >= min_struct and len(m_vecs) >= min_struct
    families_ok = n_hi_assigned >= min_hi
    m_solve = (m_info.get("solve") or {}) if isinstance(m_info, dict) else {}
    solver_ok = bool(m_solve.get("v2_converged"))

    recommended_pass = bool(coverage_ok and (families_ok or subspace_pass) and solver_ok)
    justification = (
        "Material structural validation uses adequate comparable spectrum coverage, "
        "high-confidence Hungarian-matched branch families and/or subspace/cluster preservation; "
        "not every individual mode must reach MAC>=0.85. "
        f"coverage_ok={coverage_ok} (baseline n={len(b_vecs)}, material n={len(m_vecs)}, "
        f"min={min_struct}); families_ok={families_ok} (n_hi_assigned={n_hi_assigned}); "
        f"subspace_pass={subspace_pass}; solver_ok={solver_ok}."
    )

    shape_shift_notes = [
        {
            "f_baseline_hz": r["f_baseline_hz"],
            "f_material_hz": r["f_material_hz"],
            "delta_f_hz": r["delta_f_hz"],
            "structural_MAC": r["structural_MAC"],
            "note": (
                "Substantial frequency shift with high MAC — shape-preserved branch shift, "
                "not a physics/solver failure."
            ),
        }
        for r in assigned_rows
        if r.get("shape_preserved_large_frequency_shift")
    ]

    mats = sample.get("materials") or {}
    return {
        "sample_id": material_id,
        "material_assignment": {
            "top_wood_id": mats.get("top_wood_id"),
            "back_wood_id": mats.get("back_wood_id"),
        },
        "harvest_band_hz": [band_lo, band_hi],
        "n_baseline_structural": len(b_vecs),
        "n_material_structural": len(m_vecs),
        "mac_matrix_shape": [int(mac_mat.shape[0]), int(mac_mat.shape[1])],
        "mac_matrix": mac_mat.tolist(),
        "hungarian_assignment": assigned_rows,
        "n_high_confidence_individual": int(np.sum(mac_mat >= STRUCTURAL_MAC_CONFIDENCE_THRESHOLD)),
        "n_high_confidence_assigned": n_hi_assigned,
        "subspace_overlap": subspace,
        "subspace_pass": subspace_pass,
        "coverage_ok": coverage_ok,
        "families_ok": families_ok,
        "solver_ok": solver_ok,
        "recommended_material_structural_pass": recommended_pass,
        "validation_justification": justification,
        "shape_preserved_large_frequency_shifts": shape_shift_notes,
        "baseline_load_errors": b_info.get("load_errors") or [],
        "material_load_errors": m_info.get("load_errors") or [],
    }
