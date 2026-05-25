#!/usr/bin/env python3
"""
No-eigensolve A_up coupling-strength / structural-response audit (physical-FSI continuation).

Sweeps physical_fsi_alpha on A_up/A_pu/M_pu with the validated reduced active-pressure operator,
using the decoupled acoustic seed at 244.391600 Hz. No SLEPc/EPS.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from mpi4py import MPI
from scipy.sparse.linalg import LinearOperator, cg

REPO_ROOT = Path(__file__).resolve().parents[5]
PHYSICS_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "FEM" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from coupled_participation_audit import _acoustic_reference_hz, _write_json
from physical_fsi_continuation_post import PILOT_CASE, PILOT_CONFIG
from physical_fsi_seed_residual_audit import (
    N_REDUCED_W_EXPECT,
    SEED_F_HZ,
    _assemble_reduced_continuation_operator,
    _block_residual_contributions,
    _load_alpha0_seed,
    _mask_on_indices,
    _petsc_matvec,
    _petsc_vec_from_array,
    _rayleigh_metrics,
    _validate_reduced_layout,
)

ALPHA_SWEEP = (0.0, 1.0e-6, 1.0e-5, 1.0e-4, 3.0e-4, 1.0e-3, 1.0e-2)
LINEAR_REL_TOL = 0.18
LINEAR_CV_TOL = 0.22
STRUCTURAL_CORRECTION_ALPHA = 1.0e-2


def _pressure_to_structure_forcing(
    A: Any,
    M: Any,
    x0: np.ndarray,
    *,
    lam0: float,
    u_idx: np.ndarray,
    p_idx: np.ndarray,
) -> Dict[str, float]:
    """Structural-row forcing from seed pressure via A_up and M_pu at lambda0."""
    n = int(x0.size)
    x_p = _mask_on_indices(n, x0, p_idx)
    vx_p = _petsc_vec_from_array(A, x_p)
    try:
        Ax_p, ay = _petsc_matvec(A, vx_p)
        Mx_p, my = _petsc_matvec(M, vx_p)
    finally:
        vx_p.destroy()
        ay.destroy()
        my.destroy()
    u_idx = np.asarray(u_idx, dtype=np.int32).ravel()
    f_up = Ax_p[u_idx]
    f_mp = -float(lam0) * Mx_p[u_idx]
    f_combined = f_up + f_mp
    return {
        "A_up_force_u_norm": float(np.linalg.norm(f_up)),
        "M_pu_force_u_norm": float(np.linalg.norm(f_mp)),
        "combined_pressure_to_structure_force_u_norm": float(np.linalg.norm(f_combined)),
    }


def _scipy_cg_compatible(
    Kop: LinearOperator,
    rhs: np.ndarray,
    *,
    rtol: float = 1.0e-8,
    atol: float = 0.0,
    maxiter: int = 1200,
) -> Tuple[np.ndarray, int, str]:
    """SciPy-version-tolerant CG: ``rtol`` (newer) then ``tol`` (older) on TypeError."""
    base_kw = {"atol": atol, "maxiter": maxiter}
    try:
        delta_u, info = cg(Kop, rhs, rtol=rtol, **base_kw)
        return np.asarray(delta_u, dtype=np.float64).ravel(), int(info), "rtol"
    except TypeError as exc:
        msg = str(exc).lower()
        if "rtol" not in msg and "unexpected keyword" not in msg:
            raise
        delta_u, info = cg(Kop, rhs, tol=rtol, **base_kw)
        return np.asarray(delta_u, dtype=np.float64).ravel(), int(info), "tol"


def _solve_uu_structural_correction(
    A: Any,
    M: Any,
    x0: np.ndarray,
    *,
    lam0: float,
    u_idx: np.ndarray,
    p_idx: np.ndarray,
    rhs_u: np.ndarray,
) -> Dict[str, Any]:
    """
    Solve (A_uu - lambda0 M_uu) delta_u = rhs_u on active structural DOFs (CG, no EPS).
    """
    u_idx = np.asarray(u_idx, dtype=np.int32).ravel()
    p_idx = np.asarray(p_idx, dtype=np.int32).ravel()
    n = int(x0.size)
    n_u = int(u_idx.size)

    def matvec_u(v_u: np.ndarray) -> np.ndarray:
        xw = np.zeros(n, dtype=np.float64)
        xw[u_idx] = np.asarray(v_u, dtype=np.float64).ravel()
        vx = _petsc_vec_from_array(A, xw)
        try:
            Ax, ay = _petsc_matvec(A, vx)
            Mx, my = _petsc_matvec(M, vx)
        finally:
            vx.destroy()
            ay.destroy()
            my.destroy()
        return Ax[u_idx] - float(lam0) * Mx[u_idx]

    Kop = LinearOperator((n_u, n_u), matvec=matvec_u, dtype=np.float64)
    rhs = np.asarray(rhs_u, dtype=np.float64).ravel()
    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm < 1.0e-30:
        return {
            "converged": True,
            "cg_info": 0,
            "cg_api": "skipped",
            "delta_u_norm": 0.0,
            "rhs_u_norm": rhs_norm,
            "note": "zero structural RHS; correction skipped",
        }

    try:
        delta_u, info, cg_api = _scipy_cg_compatible(
            Kop, rhs, rtol=1.0e-8, atol=0.0, maxiter=1200
        )
    except Exception as exc:
        return {
            "converged": False,
            "cg_info": -1,
            "cg_api": "failed",
            "delta_u_norm": float("nan"),
            "rhs_u_norm": rhs_norm,
            "ksp_relative_residual": float("nan"),
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "note": "uu-balance CG raised before convergence check",
        }

    converged = int(info) == 0
    rel_ksp_res = float("nan")
    if converged and rhs_norm > 0:
        Ku = matvec_u(delta_u)
        rel_ksp_res = float(np.linalg.norm(Ku - rhs) / rhs_norm)
    elif not converged:
        return {
            "converged": False,
            "cg_info": int(info),
            "cg_api": cg_api,
            "delta_u_norm": float(np.linalg.norm(delta_u)),
            "rhs_u_norm": rhs_norm,
            "ksp_relative_residual": rel_ksp_res,
            "failure_reason": (
                f"CG did not converge (info={int(info)}); operator may be singular, "
                "indefinite, or poorly conditioned"
            ),
            "note": "uu-balance correction withheld",
        }

    x_corr = np.asarray(x0, dtype=np.float64).ravel().copy()
    x_corr[u_idx] = x_corr[u_idx] + delta_u
    return {
        "converged": converged,
        "cg_info": int(info),
        "cg_api": cg_api,
        "delta_u_norm": float(np.linalg.norm(delta_u)),
        "rhs_u_norm": rhs_norm,
        "ksp_relative_residual": rel_ksp_res,
        "x_corrected": x_corr,
    }


def _analyze_alpha_scaling(
    rows: List[Dict[str, Any]],
    *,
    baseline_uu_pp_norm: float,
) -> Dict[str, Any]:
    nz = [r for r in rows if float(r["alpha_fsi"]) > 0.0]
    alphas = np.array([float(r["alpha_fsi"]) for r in nz], dtype=np.float64)
    r_up = np.array(
        [
            float(r["block_residual_contributions"]["A_up"]["residual_norm"])
            for r in nz
        ],
        dtype=np.float64,
    )
    rel_tot = np.array([float(r["relative_residual"]) for r in nz], dtype=np.float64)

    slope_origin = float(np.dot(alphas, r_up) / max(np.dot(alphas, alphas), 1.0e-30))
    pred = slope_origin * alphas
    rel_err = np.abs(r_up - pred) / np.maximum(r_up, 1.0e-30)
    per_alpha_ratio = r_up / alphas
    cv_ratio = (
        float(np.std(per_alpha_ratio) / np.mean(per_alpha_ratio))
        if per_alpha_ratio.size and np.mean(per_alpha_ratio) > 0
        else float("nan")
    )

    alpha_eq = (
        float(baseline_uu_pp_norm / slope_origin)
        if slope_origin > 1.0e-30
        else float("inf")
    )
    rel_eq = (
        float(baseline_uu_pp_norm / max(float(rows[0]["residual_norm"]), 1.0e-30))
        if rows
        else float("nan")
    )

    return {
        "A_up_residual_norm_vs_alpha": {
            str(r["alpha_fsi"]): float(
                r["block_residual_contributions"]["A_up"]["residual_norm"]
            )
            for r in rows
        },
        "linear_fit_through_origin": {
            "slope_A_up_norm_per_alpha": slope_origin,
            "max_relative_fit_error": float(np.max(rel_err)) if rel_err.size else float("nan"),
            "mean_relative_fit_error": float(np.mean(rel_err)) if rel_err.size else float("nan"),
            "per_alpha_r_up_over_alpha": {
                str(r["alpha_fsi"]): float(
                    r["block_residual_contributions"]["A_up"]["residual_norm"]
                )
                / float(r["alpha_fsi"])
                for r in nz
            },
            "coefficient_of_variation_r_up_over_alpha": cv_ratio,
        },
        "alpha_at_which_A_up_residual_norm_equals_baseline_uu_pp": alpha_eq,
        "alpha_at_which_total_relative_residual_doubles_baseline": rel_eq,
        "baseline_uu_pp_residual_norm": float(baseline_uu_pp_norm),
        "relative_residual_vs_alpha": {str(r["alpha_fsi"]): float(r["relative_residual"]) for r in rows},
        "A_up_dominates_at_alpha_1e-2": bool(
            nz
            and float(nz[-1]["block_residual_contributions"]["A_up"]["fraction_of_total_residual_norm"])
            > 0.85
        ),
    }


def _structural_response_audit(
    cfg_base: dict,
    config_path: Path,
    x0: np.ndarray,
    *,
    lam0: float,
    alpha_audit: float,
    baseline_rel: float,
) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "alpha_fsi": float(alpha_audit),
        "baseline_alpha0_relative_residual": float(baseline_rel),
        "status": "ok",
    }
    A = M = None
    try:
        A, M, _cfg, u_map, p_map, _restr = _assemble_reduced_continuation_operator(
            cfg_base,
            config_path,
            alpha_fsi=alpha_audit,
            sorting_subdir="sorting_aup_structural_response",
        )
        forcing = _pressure_to_structure_forcing(
            A, M, x0, lam0=lam0, u_idx=u_map, p_idx=p_map
        )
        n = int(x0.size)
        x_p = _mask_on_indices(n, x0, p_map)
        vx_p = _petsc_vec_from_array(A, x_p)
        try:
            Ax_p, _ = _petsc_matvec(A, vx_p)
            Mx_p, _ = _petsc_matvec(M, vx_p)
        finally:
            vx_p.destroy()
        u_map = np.asarray(u_map, dtype=np.int32).ravel()
        rhs_u = -(Ax_p[u_map] - float(lam0) * Mx_p[u_map])

        before = _block_residual_contributions(
            A, M, x0, lam0=lam0, u_idx=u_map, p_idx=p_map
        )
        solve = _solve_uu_structural_correction(
            A,
            M,
            x0,
            lam0=lam0,
            u_idx=u_map,
            p_idx=p_map,
            rhs_u=rhs_u,
        )
        rel_before = float(before["relative_residual"])
        base.update(
            {
                "pressure_to_structure_forcing": forcing,
                "relative_residual_before_correction": rel_before,
                "uu_balance_solve": solve,
                "block_A_up_fraction_before": float(
                    before["block_residual_contributions"]["A_up"][
                        "fraction_of_total_residual_norm"
                    ]
                ),
            }
        )
        if solve.get("converged") and "x_corrected" in solve:
            x_corr = solve.pop("x_corrected")
            after = _block_residual_contributions(
                A, M, x_corr, lam0=lam0, u_idx=u_map, p_idx=p_map
            )
            rel_after = float(after["relative_residual"])
            base.update(
                {
                    "relative_residual_after_correction": rel_after,
                    "residual_reduction_factor": rel_before / max(rel_after, 1.0e-30),
                    "block_A_up_fraction_after": float(
                        after["block_residual_contributions"]["A_up"][
                            "fraction_of_total_residual_norm"
                        ]
                    ),
                }
            )
        else:
            base["status"] = "uu_balance_not_converged"
            base["relative_residual_after_correction"] = float("nan")
            base["residual_reduction_factor"] = float("nan")
            base["correction_note"] = solve.get(
                "failure_reason",
                solve.get("note", "uu-balance CG did not converge"),
            )
        return base
    except Exception as exc:
        return {
            **base,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "uu_balance_solve": {"converged": False, "cg_api": "failed"},
            "correction_note": "structural-response stage failed; alpha sweep retained",
        }
    finally:
        if A is not None:
            try:
                A.destroy()
            except Exception:
                pass
        if M is not None:
            try:
                M.destroy()
            except Exception:
                pass


def _diagnostic_verdict(
    scaling: Dict[str, Any],
    structural: Dict[str, Any],
    *,
    baseline_rel: float,
) -> Dict[str, Any]:
    fit = scaling.get("linear_fit_through_origin") or {}
    max_fit_err = float(fit.get("max_relative_fit_error", float("nan")))
    cv_ratio = float(fit.get("coefficient_of_variation_r_up_over_alpha", float("nan")))
    outcomes: List[str] = []

    linear_ok = (
        math.isfinite(max_fit_err)
        and max_fit_err <= LINEAR_REL_TOL
        and math.isfinite(cv_ratio)
        and cv_ratio <= LINEAR_CV_TOL
    )
    if linear_ok and scaling.get("A_up_dominates_at_alpha_1e-2"):
        outcomes.append("A_UP_SCALING_DOMINATES_SEED_PERTURBATION")
    else:
        outcomes.append("A_UP_NONLINEAR_OR_LAYOUT_SUSPECT")

    solve = structural.get("uu_balance_solve") or {}
    rel_after = float(structural.get("relative_residual_after_correction", float("nan")))
    rel_before = float(structural.get("relative_residual_before_correction", float("nan")))
    if (
        bool(solve.get("converged"))
        and math.isfinite(rel_after)
        and rel_after < 0.5 * rel_before
        and rel_after <= max(3.0 * baseline_rel, 2.0e-2)
    ):
        outcomes.append("STRUCTURAL_RESPONSE_BALANCES_PRESSURE_FORCE_PLAUSIBLY")
    elif structural.get("status") != "ok" or not solve.get("converged"):
        if not structural.get("correction_note"):
            structural["correction_note"] = (
                "uu-balance CG did not converge or structural stage failed; "
                "structural-balance verdict withheld"
            )

    note_parts = [
        f"A_up linear-in-alpha fit max rel err={max_fit_err:.3f}, CV(r_up/alpha)={cv_ratio:.3f}",
        f"alpha_eq(A_up=uu_pp)≈{scaling.get('alpha_at_which_A_up_residual_norm_equals_baseline_uu_pp', float('nan')):.4e}",
    ]
    if "STRUCTURAL_RESPONSE_BALANCES_PRESSURE_FORCE_PLAUSIBLY" in outcomes:
        note_parts.append(
            f"uu correction at alpha={structural.get('alpha_fsi')} reduced rel residual "
            f"{rel_before:.4e}→{rel_after:.4e}"
        )
    else:
        note_parts.append(
            "pressure-only seed: Rayleigh may stay fixed while A_up residual grows with alpha"
        )

    return {
        "outcomes": outcomes,
        "primary_outcome": outcomes[0],
        "note": "; ".join(note_parts),
        "no_eps_st_or_nitsche_verdict": True,
    }


def _evaluate_alpha(
    cfg_base: dict,
    config_path: Path,
    *,
    alpha_fsi: float,
    x0: np.ndarray,
    lam0: float,
    fsi_gain: float,
) -> Dict[str, Any]:
    tag = f"sorting_aup_coupling_{alpha_fsi:.6g}".replace("+", "p")
    A, M, cfg, u_to_W, p_to_W, restr = _assemble_reduced_continuation_operator(
        cfg_base,
        config_path,
        alpha_fsi=alpha_fsi,
        sorting_subdir=tag,
    )
    _validate_reduced_layout(
        A, u_to_W, p_to_W, restr, seed_length=int(x0.size), alpha_fsi=alpha_fsi
    )
    residual = _block_residual_contributions(
        A, M, x0, lam0=lam0, u_idx=u_to_W, p_idx=p_to_W
    )
    rayleigh = _rayleigh_metrics(A, M, x0, seed_f_hz=SEED_F_HZ)
    eff_gain = float(alpha_fsi) * float(fsi_gain)
    blocks = residual["block_residual_contributions"]
    row = {
        "alpha_fsi": float(alpha_fsi),
        "effective_coupling_gain_alpha_times_fsi_coupling_gain": eff_gain,
        "fsi_coupling_gain": float(fsi_gain),
        **residual,
        **rayleigh,
        "block_residual_contributions": blocks,
    }
    try:
        A.destroy()
        M.destroy()
    except Exception:
        pass
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description="No-eigensolve A_up coupling-strength audit"
    )
    parser.add_argument("--config", type=Path, default=PILOT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=PILOT_CASE / "diagnostics")
    args = parser.parse_args()

    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[physical_fsi_aup_coupling_audit] Requires mpiexec -n 1", file=sys.stderr)
        return 2

    config_path = args.config.resolve()
    cfg_base = json.loads(config_path.read_text(encoding="utf-8"))
    fsi_gain = float(cfg_base.get("solver", {}).get("fsi_coupling_gain", 1.0e6))
    acoustic_ref = _acoustic_reference_hz(cfg_base.get("solver", {}))
    lam0 = (2.0 * math.pi * SEED_F_HZ) ** 2
    target_hz = float(cfg_base.get("solver", {}).get("_worker_target_hz", 244.39))

    x0, seed_meta = _load_alpha0_seed(target_hz, N_REDUCED_W_EXPECT)
    if MPI.COMM_WORLD.rank == 0:
        print(
            "[physical_fsi_aup_coupling_audit] "
            f"seed_f_hz={SEED_F_HZ:.6f} lambda0={lam0:.6e} len={x0.size} "
            f"alpha_sweep={ALPHA_SWEEP}",
            flush=True,
        )

    sweep_rows: List[Dict[str, Any]] = []
    for alpha in ALPHA_SWEEP:
        if MPI.COMM_WORLD.rank == 0:
            print(
                f"[physical_fsi_aup_coupling_audit] evaluating alpha_fsi={alpha:.6g}",
                flush=True,
            )
        sweep_rows.append(
            _evaluate_alpha(
                cfg_base,
                config_path,
                alpha_fsi=alpha,
                x0=x0,
                lam0=lam0,
                fsi_gain=fsi_gain,
            )
        )

    baseline_row = sweep_rows[0]
    baseline_uu = float(
        baseline_row["block_residual_contributions"]["uu_pp"]["residual_norm"]
    )
    baseline_rel = float(baseline_row["relative_residual"])
    scaling = _analyze_alpha_scaling(sweep_rows, baseline_uu_pp_norm=baseline_uu)

    try:
        structural = _structural_response_audit(
            cfg_base,
            config_path,
            x0,
            lam0=lam0,
            alpha_audit=STRUCTURAL_CORRECTION_ALPHA,
            baseline_rel=baseline_rel,
        )
    except Exception as exc:
        structural = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "alpha_fsi": float(STRUCTURAL_CORRECTION_ALPHA),
            "uu_balance_solve": {"converged": False, "cg_api": "failed"},
            "correction_note": "structural-response stage failed; alpha sweep retained",
        }
        if MPI.COMM_WORLD.rank == 0:
            print(
                "[physical_fsi_aup_coupling_audit][warn] structural stage failed: "
                f"{structural['error']}",
                flush=True,
            )

    verdict = _diagnostic_verdict(scaling, structural, baseline_rel=baseline_rel)

    report: Dict[str, Any] = {
        "experiment": "physical_fsi_aup_coupling_audit",
        "no_eigensolve": True,
        "acoustic_reference_hz": acoustic_ref,
        "seed": seed_meta,
        "lambda0_rad2_s2": lam0,
        "alpha_sweep": list(ALPHA_SWEEP),
        "sweep_results": sweep_rows,
        "A_up_scaling_analysis": scaling,
        "structural_response_audit": structural,
        "diagnostic_verdict": verdict,
        "interpretation_notes": [
            "Identical reduced maps assumed from continuation replay (validated in seed residual audit).",
            "Pressure-only seed: A_up residual can grow ~linearly with alpha while Rayleigh stays near 244.39 Hz.",
            "Do not interpret A_pu/M_pu zero fractions at pressure-only seed as global exoneration.",
            "Do not proceed to Nitsche isolation from this audit alone.",
        ],
    }

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "physical_fsi_aup_coupling_audit.json"
    _write_json(json_path, report)

    if MPI.COMM_WORLD.rank == 0:
        md = [
            "# Physical-FSI A_up coupling-strength audit (no eigensolve)",
            "",
            f"**Primary outcome:** `{verdict['primary_outcome']}`",
            "",
            f"Outcomes: {', '.join(verdict['outcomes'])}",
            "",
            verdict.get("note", ""),
            "",
            f"Seed: `{seed_meta['seed_vector_path']}` @ {SEED_F_HZ:.6f} Hz",
            "",
            "## Alpha sweep",
            "",
            "| alpha | eff. gain | rel. residual | A_up frac | rayleigh Hz |",
            "|------:|----------:|--------------:|----------:|------------:|",
        ]
        for r in sweep_rows:
            a = float(r["alpha_fsi"])
            eff = float(r["effective_coupling_gain_alpha_times_fsi_coupling_gain"])
            rel = float(r["relative_residual"])
            frac = float(
                r["block_residual_contributions"]["A_up"]["fraction_of_total_residual_norm"]
            )
            fhz = float(r["rayleigh_f_hz"])
            md.append(f"| {a:.1e} | {eff:.3e} | {rel:.4e} | {frac:.4f} | {fhz:.6f} |")
        md.extend(
            [
                "",
                "## Scaling",
                f"- alpha_eq (A_up norm = baseline uu_pp) ≈ "
                f"{scaling['alpha_at_which_A_up_residual_norm_equals_baseline_uu_pp']:.4e}",
                "",
                "## Structural response",
            ]
        )
        if structural.get("status") == "ok" and structural.get("pressure_to_structure_forcing"):
            forcing = structural["pressure_to_structure_forcing"]
            md.append(
                f"- ||f_u|| (A_up p) @ alpha={STRUCTURAL_CORRECTION_ALPHA}: "
                f"{forcing['A_up_force_u_norm']:.4e}"
            )
            solve = structural.get("uu_balance_solve") or {}
            if solve.get("cg_api"):
                md.append(f"- CG API path: `{solve['cg_api']}`")
            md.append(
                f"- rel. residual before/after uu correction: "
                f"{structural.get('relative_residual_before_correction', float('nan')):.4e} / "
                f"{structural.get('relative_residual_after_correction', float('nan')):.4e}"
            )
        else:
            md.append(
                f"- structural stage: `{structural.get('status', 'unknown')}` "
                f"({structural.get('correction_note', structural.get('error', ''))})"
            )
        md.append("")
        (out_dir / "physical_fsi_aup_coupling_audit.md").write_text(
            "\n".join(md) + "\n", encoding="utf-8"
        )
        print(f"[physical_fsi_aup_coupling_audit] outcomes={verdict['outcomes']}")
        print(f"[physical_fsi_aup_coupling_audit] wrote {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
