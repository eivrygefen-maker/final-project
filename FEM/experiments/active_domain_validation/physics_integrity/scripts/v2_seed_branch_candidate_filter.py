#!/usr/bin/env python3
"""
Experiment-only physical eligibility filters for seed-branch recovery diagnostics.

Does not alter production sigma/harvest policy. Used by diagnostic evaluate paths and
report-only candidate audits.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from physical_fsi_seed_residual_audit import (
    _block_residual_contributions,
    _rayleigh_metrics,
)

# Documented tolerances (diagnostic-only acceptance).
FILTER_POLICY: Dict[str, Any] = {
    "lambda_one_abs_tolerance": 1.0e-3,
    "reported_vs_replay_rayleigh_frequency_max_relative": 0.01,
    "reported_vs_replay_rayleigh_frequency_max_absolute_hz": 0.5,
    "replay_relative_residual_max": 0.05,
    "pressure_mac_to_true_seed_min": 0.85,
    "seed_frequency_match_max_relative": 0.01,
    "description": (
        "Physically eligible candidates must have finite reported and replay Rayleigh "
        "frequencies, agree within relative/absolute Hz tolerances, small replay residual "
        "at replay Rayleigh lambda, not be algebraic_lambda_one_suspect, and (for branch "
        "recovery) meet MAC/frequency gates."
    ),
}

VERDICT_SPURIOUS_SELECTED = (
    "DIAGNOSTIC_SELECTED_SIGMA_OR_BC_SPURIOUS_MODE_"
    "TRUE_ACOUSTIC_SEED_REMAINS_VALID_BRANCH_NOT_YET_RECOVERED"
)
VERDICT_ST_REGULARIZATION_REQUIRED = (
    "DIAGNOSTIC_ST_REGULARIZATION_REQUIRED_NO_PHYSICAL_VERDICT"
)
ST_REG_TOL = 1.0e-15


def st_operator_consistent_with_replay(
    *,
    st_a_shift_frac: float,
    st_mass_reg_frac: float,
) -> bool:
    return bool(
        abs(float(st_a_shift_frac)) <= ST_REG_TOL and abs(float(st_mass_reg_frac)) <= ST_REG_TOL
    )


def extract_st_operator_fields(solve_result: Dict[str, Any]) -> Dict[str, Any]:
    eps = solve_result.get("eps_batch_diagnostics") or {}
    diag = solve_result.get("seed_branch_recovery_diagnostic") or {}
    a_shift = float(
        diag.get(
            "actual_st_a_shift_frac",
            eps.get("st_a_shift_frac_used", solve_result.get("actual_st_a_shift_frac", 0.0)),
        )
        or 0.0
    )
    mass = float(
        diag.get(
            "actual_st_mass_reg_frac",
            eps.get("st_mass_reg_frac_used", solve_result.get("actual_st_mass_reg_frac", 0.0)),
        )
        or 0.0
    )
    sigma = float(
        diag.get(
            "actual_sigma_hz",
            eps.get("st_sigma_hz_used", solve_result.get("st_sigma_hz_used", float("nan"))),
        )
    )
    consistent = bool(
        diag.get(
            "diagnostic_operator_consistent_with_replay",
            solve_result.get("diagnostic_operator_consistent_with_replay"),
        )
    )
    if consistent is None:
        consistent = st_operator_consistent_with_replay(
            st_a_shift_frac=a_shift, st_mass_reg_frac=mass
        )
    return {
        "diagnostic_requires_unregularized_ST": bool(
            diag.get("diagnostic_requires_unregularized_ST")
            or eps.get("diagnostic_requires_unregularized_ST")
        ),
        "actual_sigma_hz": sigma,
        "actual_st_a_shift_frac": a_shift,
        "actual_st_mass_reg_frac": mass,
        "diagnostic_operator_consistent_with_replay": bool(consistent),
    }


def algebraic_lambda_one_suspect(lam_r: float, *, tol: Optional[float] = None) -> bool:
    t = float(tol if tol is not None else FILTER_POLICY["lambda_one_abs_tolerance"])
    return math.isfinite(lam_r) and abs(float(lam_r) - 1.0) <= t


def reported_vs_replay_frequency_consistent(
    reported_f_hz: float,
    replay_f_hz: float,
    *,
    seed_f_hz: Optional[float] = None,
) -> bool:
    if not (math.isfinite(reported_f_hz) and math.isfinite(replay_f_hz)):
        return False
    if replay_f_hz <= 0.0:
        return False
    rel_tol = float(FILTER_POLICY["reported_vs_replay_rayleigh_frequency_max_relative"])
    abs_tol = float(FILTER_POLICY["reported_vs_replay_rayleigh_frequency_max_absolute_hz"])
    rel_err = abs(reported_f_hz - replay_f_hz) / replay_f_hz
    if rel_err <= rel_tol:
        return True
    if abs(reported_f_hz - replay_f_hz) <= abs_tol:
        return True
    return False


def replay_candidate_metrics(
    A,
    M,
    vec: np.ndarray,
    *,
    u_to_W: np.ndarray,
    p_to_W: np.ndarray,
    reported_f_hz: float,
) -> Dict[str, Any]:
    rayleigh = _rayleigh_metrics(A, M, vec, seed_f_hz=reported_f_hz)
    lam_r = float(rayleigh["rayleigh_lambda"])
    replay_f = float(rayleigh["rayleigh_f_hz"])
    residual = _block_residual_contributions(
        A, M, vec, lam0=lam_r, u_idx=u_to_W, p_idx=p_to_W
    )
    rel_res = float(residual["relative_residual"])
    lam_one = algebraic_lambda_one_suspect(lam_r)
    freq_ok = reported_vs_replay_frequency_consistent(reported_f_hz, replay_f)
    return {
        "replay_rayleigh_lambda": lam_r,
        "replay_rayleigh_frequency_hz": replay_f,
        "replay_relative_residual": rel_res,
        "algebraic_lambda_one_suspect": lam_one,
        "reported_vs_replay_frequency_consistent": freq_ok,
        "reported_frequency_hz": float(reported_f_hz),
    }


def assess_physical_eligibility(
    *,
    reported_f_hz: float,
    replay_metrics: Dict[str, Any],
    pressure_mac_to_true_seed: float,
    seed_f_hz: Optional[float] = None,
    require_mac: bool = False,
    require_seed_frequency_match: bool = False,
) -> Dict[str, Any]:
    """Return eligibility flags and rejection reasons (diagnostic-only)."""
    replay_f = float(replay_metrics.get("replay_rayleigh_frequency_hz", float("nan")))
    rel_res = float(replay_metrics.get("replay_relative_residual", float("nan")))
    lam_r = float(replay_metrics.get("replay_rayleigh_lambda", float("nan")))
    lam_one = bool(replay_metrics.get("algebraic_lambda_one_suspect", False))
    freq_ok = bool(replay_metrics.get("reported_vs_replay_frequency_consistent", False))

    reasons: list[str] = []
    if not math.isfinite(reported_f_hz):
        reasons.append("reported_frequency_not_finite")
    if not math.isfinite(replay_f):
        reasons.append("replay_rayleigh_frequency_not_finite")
    if lam_one:
        reasons.append("algebraic_lambda_one_suspect")
    if not freq_ok:
        reasons.append("reported_vs_replay_rayleigh_frequency_mismatch")
    if not math.isfinite(rel_res) or rel_res > float(FILTER_POLICY["replay_relative_residual_max"]):
        reasons.append("replay_residual_too_large")
    if require_mac:
        mac_min = float(FILTER_POLICY["pressure_mac_to_true_seed_min"])
        if not math.isfinite(pressure_mac_to_true_seed) or pressure_mac_to_true_seed < mac_min:
            reasons.append("pressure_mac_below_threshold")
    if require_seed_frequency_match and seed_f_hz is not None and math.isfinite(seed_f_hz):
        d_frac = abs(reported_f_hz - seed_f_hz) / seed_f_hz if seed_f_hz > 0 else float("inf")
        if d_frac > float(FILTER_POLICY["seed_frequency_match_max_relative"]):
            reasons.append("reported_frequency_outside_seed_band")
        if math.isfinite(replay_f):
            d_replay = abs(replay_f - seed_f_hz) / seed_f_hz if seed_f_hz > 0 else float("inf")
            if d_replay > float(FILTER_POLICY["seed_frequency_match_max_relative"]):
                reasons.append("replay_frequency_outside_seed_band")

    physically_eligible = len(reasons) == 0
    recovers_branch = bool(
        physically_eligible and require_mac and require_seed_frequency_match
    )

    return {
        "physically_eligible_after_filter": physically_eligible,
        "rejection_reasons": reasons,
        "recovers_true_seed_branch": recovers_branch,
        "branch_recovery_pass": recovers_branch,
        "replay_rayleigh_lambda": lam_r,
        "replay_rayleigh_frequency_hz": replay_f,
        "replay_relative_residual": rel_res,
        "algebraic_lambda_one_suspect": lam_one,
        "reported_vs_replay_frequency_consistent": freq_ok,
    }


def branch_recovery_from_row(row: Dict[str, Any]) -> bool:
    """Strict branch-recovery acceptance (diagnostic-only)."""
    if not bool(row.get("continuation_seed_applied", True)):
        return False
    st_ok = st_operator_consistent_with_replay(
        st_a_shift_frac=float(row.get("actual_st_a_shift_frac", 0.0) or 0.0),
        st_mass_reg_frac=float(row.get("actual_st_mass_reg_frac", 0.0) or 0.0),
    )
    op_ok = bool(row.get("diagnostic_operator_consistent_with_replay", st_ok)) and st_ok
    if not op_ok:
        return False
    if bool(row.get("algebraic_lambda_one_suspect", False)):
        return False
    if not bool(row.get("reported_vs_replay_frequency_consistent", False)):
        return False
    return bool(
        row.get("branch_recovery_pass")
        or (
            row.get("physically_eligible_after_filter")
            and row.get("recovers_true_seed_branch")
        )
    )


VERDICT_FILTERED_BRANCH_RECOVERED = "FILTERED_DIAGNOSTIC_BRANCH_RECOVERED"
VERDICT_FILTERED_NO_BRANCH = "FILTERED_DIAGNOSTIC_NO_PHYSICAL_BRANCH_RECOVERED"
VERDICT_FILTERED_INCONSISTENT = "FILTERED_DIAGNOSTIC_OUTPUT_OR_REPLAY_INCONSISTENT"


def assign_filtered_evaluation_verdict(
    candidates: list[Dict[str, Any]],
    *,
    artifacts_ok: bool,
    expected_mode_count: Optional[int] = None,
) -> str:
    if not artifacts_ok:
        return VERDICT_FILTERED_INCONSISTENT
    if expected_mode_count is not None and len(candidates) != int(expected_mode_count):
        return VERDICT_FILTERED_INCONSISTENT
    if any(branch_recovery_from_row(c) for c in candidates):
        return VERDICT_FILTERED_BRANCH_RECOVERED
    if candidates:
        return VERDICT_FILTERED_NO_BRANCH
    return VERDICT_FILTERED_INCONSISTENT
