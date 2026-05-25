#!/usr/bin/env python3
"""
Experiment-only mode diagnostics: raw vs GNHEP-back-transformed pressure participation.

Does not modify production harvest or p_frac computation.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[5]  # repo root (…/final-project)
SCRIPT_DIR = REPO_ROOT / "FEM" / "scripts"
import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fem_main_3d as fem3d
from fem_mode_array_utils import MODE_VECTOR_FILE_SUFFIX, load_mode_column_any


def parse_gnhep_scales_from_log(log_path: Path) -> Dict[str, float]:
    """Parse GNHEP block scales and pressure_dof_scale from solver stdout."""
    out: Dict[str, float] = {}
    if not log_path.is_file():
        return out
    text = log_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(
        r"GNHEP block Frobenius scales:\s*s_uu=([\d.eE+-]+),\s*s_pp=([\d.eE+-]+),\s*s_couple=([\d.eE+-]+)",
        text,
    )
    if m:
        out["s_uu"] = float(m.group(1))
        out["s_pp"] = float(m.group(2))
        out["s_couple"] = float(m.group(3))
    m2 = re.search(r"coupled pressure_dof_scale=([\d.eE+-]+)", text)
    if m2:
        out["pressure_dof_scale"] = float(m2.group(1))
    m3 = re.search(r"fsi_coupling_gain=([\d.eE+-]+)", text)
    if m3:
        out["fsi_coupling_gain"] = float(m3.group(1))
    return out


def merge_scaling_metadata(
    case_dir: Path,
    result_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Prefer captured _physics_integrity from solve; fall back to log parsing."""
    meta: Dict[str, float] = {}
    audit_path = case_dir / "diagnostics" / "physics_integrity_audit.json"
    if audit_path.is_file():
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            gn = audit.get("gnhep_scales") or {}
            for k in ("s_uu", "s_pp", "s_couple", "gnhep_global"):
                if k in gn:
                    meta[k] = float(gn[k])
            if "pressure_dof_scale" in audit:
                meta["pressure_dof_scale"] = float(audit["pressure_dof_scale"])
        except Exception:
            pass
    if not meta.get("s_uu") or not meta.get("s_pp"):
        logs = sorted((case_dir / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if logs:
            meta.update(parse_gnhep_scales_from_log(logs[0]))
    if result_json:
        op = result_json.get("operator_meta") or {}
        pi = op.get("physics_integrity") or result_json.get("physics_integrity") or {}
        gn = pi.get("gnhep_scales") or pi
        for k in ("s_uu", "s_pp", "s_couple"):
            if k in gn and k not in meta:
                meta[k] = float(gn[k])
    meta.setdefault("s_uu", 1.0)
    meta.setdefault("s_pp", 1.0)
    meta.setdefault("s_couple", math.sqrt(meta["s_uu"] * meta["s_pp"]))
    meta.setdefault("pressure_dof_scale", 30.0)
    return meta


def _max_block_abs(arr: np.ndarray, idx: np.ndarray) -> float:
    if idx.size == 0:
        return 0.0
    vals = np.asarray(arr[np.asarray(idx, dtype=np.int32)], dtype=np.float64)
    return float(np.max(np.abs(vals))) if vals.size else 0.0


def classify_mode(
    *,
    p_frac_phys: float,
    wood: float,
    p_block_max_phys: float,
) -> str:
    if p_frac_phys >= 0.35 and wood < 0.25:
        return "acoustic_dominated"
    if p_frac_phys >= 0.05 and wood >= 0.15:
        return "coupled"
    if wood >= 0.5 and p_frac_phys < 1.0e-4:
        return "structural_dominated"
    if p_frac_phys < 1.0e-5:
        return "structural_dominated"
    return "mixed"


def diagnose_mixed_mode(
    arr: np.ndarray,
    *,
    u_to_W: np.ndarray,
    p_to_W: np.ndarray,
    gnhep: Dict[str, float],
    wood_top: float = 0.0,
    wood_back: float = 0.0,
    frequency_hz: float = float("nan"),
    p_exclude_rows: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Per-mode metrics in raw SLEPc coords and GNHEP-back-transformed physical coords."""
    s_u = max(float(gnhep.get("s_uu", 1.0)), 1.0e-30)
    s_p = max(float(gnhep.get("s_pp", 1.0)), 1.0e-30)
    u_idx = np.asarray(u_to_W, dtype=np.int32).ravel()
    p_idx = np.asarray(p_to_W, dtype=np.int32).ravel()
    if p_exclude_rows is not None and p_exclude_rows.size > 0:
        excl = np.unique(np.asarray(p_exclude_rows, dtype=np.int32).ravel())
        p_idx = p_idx[~np.isin(p_idx, excl)]

    u_n_raw, p_n_raw = fem3d._mixed_eigenvector_block_norms(
        arr, u_to_W=u_idx, p_to_W=p_idx
    )
    p_frac_raw = p_n_raw / max(u_n_raw + p_n_raw, 1.0e-30)
    p_max_raw = _max_block_abs(arr, p_idx)

    arr_phys = np.asarray(arr, dtype=np.float64).copy()
    if u_idx.size:
        arr_phys[u_idx] *= s_u
    if p_idx.size:
        arr_phys[p_idx] *= s_p
    u_n_phys, p_n_phys = fem3d._mixed_eigenvector_block_norms(
        arr_phys, u_to_W=u_idx, p_to_W=p_idx
    )
    p_frac_phys = p_n_phys / max(u_n_phys + p_n_phys, 1.0e-30)
    p_max_phys = _max_block_abs(arr_phys, p_idx)

    wood = float(wood_top) + float(wood_back)
    mode_class = classify_mode(
        p_frac_phys=p_frac_phys,
        wood=wood,
        p_block_max_phys=p_max_phys,
    )
    ratio_phys = p_n_phys / max(u_n_phys, 1.0e-30)
    return {
        "frequency_hz": float(frequency_hz),
        "p_frac_raw": float(p_frac_raw),
        "p_frac_phys_gnhep": float(p_frac_phys),
        "p_over_u_phys": float(ratio_phys),
        "wood_participation": wood,
        "top_plate_frac": float(wood_top),
        "back_plate_frac": float(wood_back),
        "u_norm_raw": float(u_n_raw),
        "p_norm_raw": float(p_n_raw),
        "u_norm_phys_gnhep": float(u_n_phys),
        "p_norm_phys_gnhep": float(p_n_phys),
        "p_block_max_raw": float(p_max_raw),
        "p_block_max_phys_gnhep": float(p_max_phys),
        "mode_class": mode_class,
        "gnhep_s_uu": s_u,
        "gnhep_s_pp": s_p,
        "pressure_dof_scale": float(gnhep.get("pressure_dof_scale", 30.0)),
    }


def diagnose_pressure_only_mode(
    arr: np.ndarray,
    *,
    gnhep: Dict[str, float],
    frequency_hz: float = float("nan"),
) -> Dict[str, Any]:
    s_p = max(float(gnhep.get("s_pp", 1.0)), 1.0e-30)
    p_n_raw = float(np.linalg.norm(arr))
    arr_phys = np.asarray(arr, dtype=np.float64) * s_p
    p_n_phys = float(np.linalg.norm(arr_phys))
    p_max_raw = float(np.max(np.abs(arr))) if arr.size else 0.0
    p_max_phys = float(np.max(np.abs(arr_phys))) if arr_phys.size else 0.0
    return {
        "frequency_hz": float(frequency_hz),
        "p_frac_raw": 1.0,
        "p_frac_phys_gnhep": 1.0,
        "p_over_u_phys": float("inf"),
        "wood_participation": 0.0,
        "top_plate_frac": 0.0,
        "back_plate_frac": 0.0,
        "u_norm_raw": 0.0,
        "p_norm_raw": p_n_raw,
        "u_norm_phys_gnhep": 0.0,
        "p_norm_phys_gnhep": p_n_phys,
        "p_block_max_raw": p_max_raw,
        "p_block_max_phys_gnhep": p_max_phys,
        "mode_class": "acoustic_dominated",
        "gnhep_s_uu": float(gnhep.get("s_uu", 1.0)),
        "gnhep_s_pp": s_p,
        "pressure_dof_scale": float(gnhep.get("pressure_dof_scale", 30.0)),
    }


def diagnose_structural_mode(
    arr: np.ndarray,
    *,
    wood_top: float,
    wood_back: float,
    frequency_hz: float = float("nan"),
    gnhep: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    gn = gnhep or {"s_uu": 1.0, "s_pp": 1.0}
    s_u = max(float(gn.get("s_uu", 1.0)), 1.0e-30)
    u_n_raw = float(np.linalg.norm(arr))
    arr_phys = np.asarray(arr, dtype=np.float64) * s_u
    u_n_phys = float(np.linalg.norm(arr_phys))
    wood = float(wood_top) + float(wood_back)
    return {
        "frequency_hz": float(frequency_hz),
        "p_frac_raw": 0.0,
        "p_frac_phys_gnhep": 0.0,
        "p_over_u_phys": 0.0,
        "wood_participation": wood,
        "top_plate_frac": float(wood_top),
        "back_plate_frac": float(wood_back),
        "u_norm_raw": u_n_raw,
        "p_norm_raw": 0.0,
        "u_norm_phys_gnhep": u_n_phys,
        "p_norm_phys_gnhep": 0.0,
        "p_block_max_raw": 0.0,
        "p_block_max_phys_gnhep": 0.0,
        "mode_class": "structural_dominated",
        "gnhep_s_uu": s_u,
        "gnhep_s_pp": float(gn.get("s_pp", 1.0)),
        "pressure_dof_scale": float(gn.get("pressure_dof_scale", 30.0)),
    }


P_FRAC_PRODUCTION_DEFINITION = (
    "Production/harvest p_frac: L2 ratio ||p||_2 / (||u||_2 + ||p||_2) on the mixed W global "
    "eigenvector returned by SLEPc, using collapse maps u_to_W and p_to_W. The vector lives in the "
    "assembled GNHEP basis after block Frobenius scaling (u,p blocks scaled by 1/s_uu, 1/s_pp) and "
    "with pressure_dof_scale baked into the p–p and u–p forms (algebraic pressure DOFs). It is NOT "
    "an energy participation metric and does not undo pressure_dof_scale."
)

P_FRAC_PHYS_GNHEP_DEFINITION = (
    "Experiment p_frac_phys_gnhep: same L2 norm ratio after multiplying u coefficients by s_uu and "
    "p coefficients by s_pp (undo block Frobenius form scaling only). pressure_dof_scale is unchanged."
)

P_FRAC_FULLY_UNSCALED_DEFINITION = (
    "Experiment p_frac_fully_unscaled: L2 ratio after u*=s_uu and p*=s_pp/p_scale (undo GNHEP block "
    "scaling and pressure_dof similarity so pressure coefficients map to physical pressure amplitude)."
)


def unscale_mixed_mode_vector(
    arr: np.ndarray,
    *,
    u_to_W: np.ndarray,
    p_to_W: np.ndarray,
    gnhep: Dict[str, float],
    undo_pressure_dof_scale: bool = True,
) -> np.ndarray:
    """Return a copy with GNHEP (and optionally pressure_dof_scale) undone on u/p blocks."""
    s_u = max(float(gnhep.get("s_uu", 1.0)), 1.0e-30)
    s_p = max(float(gnhep.get("s_pp", 1.0)), 1.0e-30)
    p_scale = max(float(gnhep.get("pressure_dof_scale", 30.0)), 1.0e-30)
    out = np.asarray(arr, dtype=np.float64).copy()
    u_idx = np.asarray(u_to_W, dtype=np.int32).ravel()
    p_idx = np.asarray(p_to_W, dtype=np.int32).ravel()
    if u_idx.size:
        out[u_idx] *= s_u
    if p_idx.size:
        out[p_idx] *= s_p
        if undo_pressure_dof_scale:
            out[p_idx] /= p_scale
    return out


def block_l2_p_fraction(
    arr: np.ndarray,
    *,
    u_to_W: np.ndarray,
    p_to_W: np.ndarray,
) -> Tuple[float, float, float]:
    """Return (p_frac, u_norm, p_norm) from block L2 norms."""
    u_n, p_n = fem3d._mixed_eigenvector_block_norms(
        arr, u_to_W=u_to_W, p_to_W=p_to_W
    )
    p_frac = p_n / max(u_n + p_n, 1.0e-30)
    return float(p_frac), float(u_n), float(p_n)


def compute_mass_energy_participation(
    arr: np.ndarray,
    M: Any,
    A: Any,
    *,
    u_to_W: np.ndarray,
    p_to_W: np.ndarray,
    gnhep: Dict[str, float],
) -> Dict[str, float]:
    """
    Quadratic energy splits using assembled GNHEP-scaled M (same basis as SLEPc modes).

    Physical energies apply block undo factors consistent with M_phys ≈ S M_gnhep S,
    S = diag(1/s_uu on u, 1/s_pp on p); pressure_dof_scale is already in m_pp/m_pu forms.
    """
    from petsc4py import PETSc

    s_u = max(float(gnhep.get("s_uu", 1.0)), 1.0e-30)
    s_p = max(float(gnhep.get("s_pp", 1.0)), 1.0e-30)
    s_c = max(float(gnhep.get("s_couple", math.sqrt(s_u * s_p))), 1.0e-30)

    u_idx = np.asarray(u_to_W, dtype=np.int32).ravel()
    p_idx = np.asarray(p_to_W, dtype=np.int32).ravel()
    x = np.asarray(arr, dtype=np.float64).ravel()

    x_vec = M.createVecRight()
    y_vec = M.createVecRight()
    ay_vec = A.createVecRight()
    try:
        x_vec.setArray(x.copy())
        try:
            x_vec.ghostUpdate(
                addv=PETSc.InsertMode.INSERT_VALUES,
                mode=PETSc.ScatterMode.FORWARD,
            )
        except Exception:
            pass
        M.mult(x_vec, y_vec)
        A.mult(x_vec, ay_vec)
        try:
            y_vec.ghostUpdate(
                addv=PETSc.InsertMode.INSERT_VALUES,
                mode=PETSc.ScatterMode.FORWARD,
            )
            ay_vec.ghostUpdate(
                addv=PETSc.InsertMode.INSERT_VALUES,
                mode=PETSc.ScatterMode.FORWARD,
            )
        except Exception:
            pass
        y = np.asarray(y_vec.getArray(readonly=True), dtype=np.float64).copy()
        ay = np.asarray(ay_vec.getArray(readonly=True), dtype=np.float64).copy()
    finally:
        x_vec.destroy()
        y_vec.destroy()
        ay_vec.destroy()

    u_v = x[u_idx] if u_idx.size else np.zeros(0, dtype=np.float64)
    p_v = x[p_idx] if p_idx.size else np.zeros(0, dtype=np.float64)
    mu_u = y[u_idx] if u_idx.size else np.zeros(0, dtype=np.float64)
    mp_p = y[p_idx] if p_idx.size else np.zeros(0, dtype=np.float64)

    e_struct_gnhep = 0.5 * float(np.dot(u_v, mu_u))
    e_air_gnhep = 0.5 * float(np.dot(p_v, mp_p))
    # u-row / p-row matvec includes block-diagonal plus FSI mass coupling.
    cross_u_from_p = float(np.dot(u_v, mu_u) - 2.0 * e_struct_gnhep)
    cross_p_from_u = float(np.dot(p_v, mp_p) - 2.0 * e_air_gnhep)
    cross_mass = 0.5 * (abs(cross_u_from_p) + abs(cross_p_from_u))

    e_struct_phys = s_u * e_struct_gnhep
    e_air_phys = s_p * e_air_gnhep
    cross_phys = s_c * cross_mass

    denom = e_struct_phys + e_air_phys + cross_phys
    p_frac_energy = e_air_phys / max(denom, 1.0e-30)

    stiff_u_load = float(np.linalg.norm(ay[u_idx])) if u_idx.size else 0.0
    stiff_p_load = float(np.linalg.norm(ay[p_idx])) if p_idx.size else 0.0
    stiff_cross = float(np.dot(u_v, ay[u_idx])) if u_idx.size else 0.0

    return {
        "structural_modal_energy_gnhep": e_struct_gnhep,
        "acoustic_modal_energy_gnhep": e_air_gnhep,
        "mass_cross_u_from_p_gnhep": cross_u_from_p,
        "mass_cross_p_from_u_gnhep": cross_p_from_u,
        "mass_cross_term_gnhep": cross_mass,
        "structural_modal_energy_phys": e_struct_phys,
        "acoustic_modal_energy_phys": e_air_phys,
        "mass_cross_term_phys": cross_phys,
        "p_frac_energy_phys": float(p_frac_energy),
        "stiffness_u_row_load_norm": stiff_u_load,
        "stiffness_p_row_load_norm": stiff_p_load,
        "stiffness_cross_u_dot_Ax_u": stiff_cross,
        "gnhep_s_uu": s_u,
        "gnhep_s_pp": s_p,
        "gnhep_s_couple": s_c,
    }


def load_mode_vector(path: Path, n_expected: int) -> np.ndarray:
    if path.suffix == ".npz" and path.name.endswith(MODE_VECTOR_FILE_SUFFIX):
        return load_mode_column_any(path)
    if path.suffix == ".npz":
        data = np.load(path)
        if "eigvec" in data:
            return np.asarray(data["eigvec"], dtype=np.float64).ravel()
        if "arr" in data:
            return np.asarray(data["arr"], dtype=np.float64).ravel()
    if path.suffix == ".smx.npz" or "smx" in path.name:
        return load_mode_column_any(path)
    raise FileNotFoundError(f"Unsupported mode file: {path}")


def write_mode_diagnostics_json(
    case_dir: Path,
    modes: List[Dict[str, Any]],
    *,
    case_label: str,
    scaling: Dict[str, float],
) -> Path:
    diag_dir = case_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    out_path = diag_dir / "mode_physics_diagnostics.json"
    payload = {
        "case": case_label,
        "scaling_metadata": scaling,
        "modes": modes,
        "notes": (
            "p_frac_raw matches production harvest (SLEPc vector in GNHEP-scaled assembled basis). "
            "p_frac_phys_gnhep multiplies u coeffs by s_uu and p coeffs by s_pp to undo block "
            "Frobenius form scaling only; pressure_dof_scale remains in the physical model."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
