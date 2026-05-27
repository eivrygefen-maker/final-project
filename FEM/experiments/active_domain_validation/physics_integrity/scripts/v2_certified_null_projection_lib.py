#!/usr/bin/env python3
"""Certified M_uu null-basis projection for lossless nullspace-projected adjudication v1."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from v2_clean_adjudication_lane import (
    OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1,
    OUT_SUBDIR_LOSSLESS_NULLSPACE_PROJECTED_ADJUDICATION_V1,
)
from v2_mesh_convergence_common import CONV_DIAG

NULL_BASIS_PREFLIGHT_JSON = (
    CONV_DIAG
    / "v2_lossless_adjudication_v1_Muu_null_basis_certification_and_projection_preflight.json"
)
PROJECTED_AUTH_JSON = (
    CONV_DIAG / "v2_lossless_nullspace_projected_adjudication_v1_eps_authorization_record.json"
)

CERTIFIED_NULL_REL_TOL = 1e-12
SVD_RCOND = 1e-10
SEED_PRESERVE_REL_TOL = 1e-10
STRATEGY_PROJECTED_V1 = "MAPPING_FIXED_UNREGULARIZED_LOSSLESS_NULLSPACE_PROJECTED_ADJUDICATION_V1"

REQUIRED_PREFLIGHT_GATES: Dict[str, Any] = {
    "certified_empirical_null_basis_dimension": 23,
    "certified_null_basis_certified": True,
    "seed_projection_preservation_pass_certified_null": True,
    "projected_existing_mass_null_family_sufficiently_suppressed_by_certified_null_basis": True,
    "recommended_future_strategy": STRATEGY_PROJECTED_V1,
    "no_new_eigensolve_executed": True,
    "additional_eps_authorized": False,
}


def _coalesce_list(*candidates: Any) -> list:
    for c in candidates:
        if isinstance(c, list):
            return c
    return []


def _atomic_load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def project_u(x_u: np.ndarray, Q: np.ndarray) -> np.ndarray:
    if Q.size == 0:
        return np.asarray(x_u, dtype=np.float64).ravel()
    x_u = np.asarray(x_u, dtype=np.float64).ravel()
    return x_u - Q @ (Q.T @ x_u)


def project_full(vec: np.ndarray, Q_u: np.ndarray, u_to_W: np.ndarray) -> np.ndarray:
    out = np.asarray(vec, dtype=np.float64).ravel().copy()
    out[np.asarray(u_to_W, dtype=np.int32)] = project_u(out[u_to_W], Q_u)
    return out


def embed_Q_in_W(Q_u: np.ndarray, u_to_W: np.ndarray, n_W: int) -> np.ndarray:
    Q_u = np.asarray(Q_u, dtype=np.float64)
    u_to_W = np.asarray(u_to_W, dtype=np.int32).ravel()
    k = int(Q_u.shape[1]) if Q_u.ndim == 2 else 0
    Q_w = np.zeros((int(n_W), k), dtype=np.float64)
    if k > 0:
        Q_w[u_to_W, :] = Q_u
    return Q_w


def orthonormalize_columns(Q_raw: np.ndarray) -> np.ndarray:
    if Q_raw.size == 0:
        return np.zeros((Q_raw.shape[0], 0), dtype=np.float64)
    Q_orth, _ = np.linalg.qr(np.asarray(Q_raw, dtype=np.float64), mode="reduced")
    return Q_orth.astype(np.float64, copy=False)


def validate_null_basis_preflight_gates(
    preflight: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    preflight = preflight if preflight is not None else _atomic_load_json(NULL_BASIS_PREFLIGHT_JSON)
    issues: List[str] = []
    if not preflight:
        return False, ["null_basis_preflight_json_missing"]
    for key, expected in REQUIRED_PREFLIGHT_GATES.items():
        actual = preflight.get(key)
        if actual != expected:
            issues.append(f"{key}: expected {expected!r}, got {actual!r}")
    dim = int(preflight.get("certified_empirical_null_basis_dimension", 0) or 0)
    if dim <= 0:
        issues.append("certified_empirical_null_basis_dimension_not_positive")
    return len(issues) == 0, issues


def build_Q_certified_null_from_prior_lossless_tree(
    *,
    prior_out_dir: Path,
    mesh_file: Path,
    sample: Dict[str, Any],
    u_to_W: np.ndarray,
    p_to_W: np.ndarray,
    operator_size: int,
    M: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build Q_certified_null from authoritative prior lossless EPS vectors."""
    from fem_mode_array_utils import load_mode_dense_f64_lossless
    from run_v2_lossless_adjudication_v1_u_active_nullspace_attribution import (
        _build_tag_subsets_in_reduced_u,
    )

    modes = _coalesce_list(
        _atomic_load_json(prior_out_dir / "diagnostics/mode_energy_summary.json").get("modes")
    )
    columns: List[np.ndarray] = []
    for m in modes:
        rel = m.get("vector_file_lossless")
        if not rel:
            continue
        lp = prior_out_dir / str(rel)
        if not lp.is_file():
            continue
        vec = np.asarray(load_mode_dense_f64_lossless(lp), dtype=np.float64).ravel()
        x_u = vec[np.asarray(u_to_W, dtype=np.int32)]
        xn = float(np.linalg.norm(x_u))
        if xn <= 0:
            continue
        columns.append((x_u / xn).astype(np.float64, copy=False))

    if not columns:
        raise RuntimeError("no prior lossless u_active vectors for certified-null basis")

    X = np.column_stack(columns)
    X_svd_u, s_vals, _ = np.linalg.svd(X, full_matrices=False)
    rank_full = int(np.sum(s_vals > SVD_RCOND * float(s_vals[0]))) if s_vals.size else 0

    def _matvec_u(op: Any, x_u: np.ndarray, u_idx: np.ndarray) -> np.ndarray:
        from physical_fsi_seed_residual_audit import _petsc_matvec, _petsc_vec_from_array

        n = int(op.getSize()[1])
        x_full = np.zeros(n, dtype=np.float64)
        x_full[np.asarray(u_idx, dtype=np.int32)] = np.asarray(x_u, dtype=np.float64).ravel()
        vx = _petsc_vec_from_array(op, x_full)
        try:
            y, vy = _petsc_matvec(op, vx)
            y_arr = np.asarray(y, dtype=np.float64).ravel()
        finally:
            vx.destroy()
            if vy is not None:
                vy.destroy()
        return y_arr[np.asarray(u_idx, dtype=np.int32)]

    certified_indices: List[int] = []
    direction_rows: List[Dict[str, Any]] = []
    for i in range(rank_full):
        q = X_svd_u[:, i]
        Mq = _matvec_u(M, q, u_to_W)
        rel_m = float(np.linalg.norm(Mq) / max(float(np.linalg.norm(q)), 1e-300))
        is_null = bool(rel_m < CERTIFIED_NULL_REL_TOL)
        if is_null:
            certified_indices.append(i)
        direction_rows.append(
            {
                "basis_index": i,
                "relative_mass_action": rel_m,
                "mass_null_direction": is_null,
            }
        )

    Q_raw = X_svd_u[:, certified_indices] if certified_indices else np.zeros((u_to_W.size, 0))
    Q_cert = orthonormalize_columns(Q_raw)

    tag_map = _build_tag_subsets_in_reduced_u(
        mesh_file=mesh_file,
        sample=sample,
        u_to_W=u_to_W,
        p_to_W=p_to_W,
        operator_size=operator_size,
    )

    meta = {
        "prior_output_subdir": OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1,
        "prior_out_dir": str(prior_out_dir),
        "candidate_vector_count": len(columns),
        "numerical_rank_full": rank_full,
        "certified_null_basis_indices": certified_indices,
        "certified_null_basis_threshold": CERTIFIED_NULL_REL_TOL,
        "certified_empirical_null_basis_dimension": int(Q_cert.shape[1]),
        "certified_null_basis_certified": bool(
            Q_cert.shape[1] > 0
            and all(direction_rows[i]["mass_null_direction"] for i in certified_indices)
        ),
        "Q_full_dimension_diagnostic_only": rank_full,
        "Q_full_used_for_solver": False,
        "independent_direction_diagnostics": direction_rows,
        "tag_map_subset_sizes": {
            k: int(np.asarray(v).size) for k, v in (tag_map.get("subsets") or {}).items()
        },
    }
    return Q_cert, meta


def verify_Q_certified_properties(
    Q: np.ndarray, *, M: Any, u_to_W: np.ndarray, seed_vec: np.ndarray
) -> Dict[str, Any]:
    def _matvec_u(op: Any, x_u: np.ndarray, u_idx: np.ndarray) -> np.ndarray:
        from physical_fsi_seed_residual_audit import _petsc_matvec, _petsc_vec_from_array

        n = int(op.getSize()[1])
        x_full = np.zeros(n, dtype=np.float64)
        x_full[np.asarray(u_idx, dtype=np.int32)] = np.asarray(x_u, dtype=np.float64).ravel()
        vx = _petsc_vec_from_array(op, x_full)
        try:
            y, vy = _petsc_matvec(op, vx)
            y_arr = np.asarray(y, dtype=np.float64).ravel()
        finally:
            vx.destroy()
            if vy is not None:
                vy.destroy()
        return y_arr[np.asarray(u_idx, dtype=np.int32)]

    k = int(Q.shape[1])
    gram = Q.T @ Q if k else np.zeros((0, 0))
    off_diag = float(np.max(np.abs(gram - np.eye(k)))) if k else 0.0
    per_dir: List[Dict[str, Any]] = []
    for j in range(k):
        q = Q[:, j]
        Mq = _matvec_u(M, q, u_to_W)
        rel_m = float(np.linalg.norm(Mq) / max(float(np.linalg.norm(q)), 1e-300))
        per_dir.append({"index": j, "relative_mass_action": rel_m, "mass_null_direction": rel_m < CERTIFIED_NULL_REL_TOL})
    seed_u = seed_vec[u_to_W]
    seed_proj = project_u(seed_u, Q)
    seed_rel = float(np.linalg.norm(seed_proj - seed_u) / max(float(np.linalg.norm(seed_vec)), 1e-300))
    return {
        "Q_shape": [int(Q.shape[0]), k],
        "Q_gram_off_diagonal_max": off_diag,
        "Q_orthonormal_within_tolerance": bool(off_diag < 1e-8),
        "per_direction_mass_null_check": per_dir,
        "all_directions_mass_null": bool(all(d["mass_null_direction"] for d in per_dir)),
        "seed_projection_relative_change_norm_ratio": seed_rel,
        "seed_projection_preservation_pass": bool(seed_rel <= SEED_PRESERVE_REL_TOL),
    }


def persist_projection_basis(
    out_dir: Path,
    Q_cert: np.ndarray,
    meta: Dict[str, Any],
) -> Path:
    diag = out_dir / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    q_path = diag / "certified_null_Q_u.npy"
    np.save(str(q_path), Q_cert)
    gate_path = diag / "certified_null_projection_gate.json"
    gate_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return q_path


def load_persisted_Q_certified(out_dir: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    q_path = out_dir / "diagnostics/certified_null_Q_u.npy"
    gate_path = out_dir / "diagnostics/certified_null_projection_gate.json"
    if not q_path.is_file():
        raise FileNotFoundError(f"missing {q_path}")
    Q = np.load(str(q_path))
    meta = _atomic_load_json(gate_path)
    return Q, meta


def orthogonality_fraction_to_Q(x_u: np.ndarray, Q: np.ndarray) -> float:
    xn = float(np.linalg.norm(x_u))
    if xn <= 0 or Q.size == 0:
        return 0.0
    return float(np.linalg.norm(Q.T @ x_u) / xn)
