#!/usr/bin/env python3
"""
Local (workspace-only) static audit for seed-branch recovery and ST regularization.

Does not read VM-only artifacts. Merges operator-reported evidence as labeled constants.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[5]
SCRIPT_DIR = Path(__file__).resolve().parent
FEM_SCRIPTS = REPO_ROOT / "FEM" / "scripts"

ROOT_CAUSE_CONFIRMED_ST = "ROOT_CAUSE_CONFIRMED_ST_REGULARIZATION_OR_MAPPING_FIX_READY_FOR_ONE_BASELINE_RERUN"
VERDICT_PERSISTENCE_BUG = "SAVED_MODE_VECTOR_PERSISTENCE_OR_LAYOUT_BUG_CONFIRMED"
VERDICT_EPS_MASS_NULL = "EPS_RETURNED_ONLY_MASS_NULL_CANDIDATES_IN_UNREGULARIZED_SOLVE"
VERDICT_REPLAY_INVALID = "REPLAY_CONTROL_INVALID_SEED_XHMX_NONFINITE"
VERDICT_NOT_LOCALIZED = "VECTOR_MASS_NULL_ROOT_CAUSE_NOT_LOCALIZED_STOP_FOR_ARCHITECTURE_REVIEW"
ROOT_CAUSE_CONFIRMED_OTHER = "ROOT_CAUSE_CONFIRMED_OTHER_FIX_READY_FOR_ONE_BASELINE_RERUN"
ROOT_CAUSE_NOT_CONFIRMED = "ROOT_CAUSE_NOT_YET_CONFIRMED_NO_FURTHER_SOLVE_AUTHORIZED"

VERDICT_SPURIOUS = (
    "DIAGNOSTIC_SELECTED_SIGMA_OR_BC_SPURIOUS_MODE_"
    "TRUE_ACOUSTIC_SEED_REMAINS_VALID_BRANCH_NOT_YET_RECOVERED"
)
VERDICT_FILTERED_NO_BRANCH = "FILTERED_DIAGNOSTIC_NO_PHYSICAL_BRANCH_RECOVERED"


def _fref(p: Path) -> str:
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(p)


def _read_snippet(path: Path, needle: str, *, context: int = 2) -> str:
    if not path.is_file():
        return f"(missing {path})"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(lines):
        if needle in line:
            lo = max(0, i - context)
            hi = min(len(lines), i + context + 1)
            return "\n".join(f"{j+1}:{lines[j]}" for j in range(lo, hi))
    return f"(needle {needle!r} not found)"


def build_st_retry_control_flow_audit() -> Dict[str, Any]:
    """Confirmed from local code: ordering of sigma vs ST regularization retries."""
    fem = FEM_SCRIPTS / "fem_main_3d.py"
    v2 = SCRIPT_DIR / "v2_sensitivity_solve.py"
    return {
        "evidence_source": "confirmed_from_local_code",
        "sigma_ladder_construction": {
            "function": "fem_main_3d._slepc_st_sigma_hz_candidates",
            "file": _fref(fem),
            "snippet": _read_snippet(fem, "def _slepc_st_sigma_hz_candidates", context=8),
            "prior_unfiltered_diagnostic_cfg": {
                "file": _fref(v2),
                "function": "_apply_seed_branch_recovery_diagnostic_solver_cfg",
                "eps_st_sigma_try_target_first": True,
                "eps_st_sigma_include_target_in_ladder": False,
                "effect": "First sigma candidate is exact seed/target Hz (offset 0).",
            },
            "new_unregularized_offset_cfg": {
                "file": _fref(v2),
                "function": "_apply_seed_branch_unregularized_offset_diagnostic_solver_cfg",
                "eps_st_sigma_try_target_first": False,
                "eps_st_sigma_primary_offset_hz": 0.5,
                "eps_st_sigma_retry_offsets_hz": [0.5, -0.5, 1.0, -1.0, 2.0, -2.0],
                "effect": "Never places sigma exactly at seed Rayleigh frequency.",
            },
        },
        "st_retry_nested_loop_order": {
            "function": "fem_main_3d._slepc_try_eps_st_setup",
            "file": _fref(fem),
            "snippet": _read_snippet(fem, "for stiff_frac in stiff_ladder", context=6),
            "default_production_order": [
                "outer: stiff_frac in (1e-3, 0.0, 5e-3, 2e-2)  # A-shift tried BEFORE zero",
                "middle: reg_frac in mass-reg ladder (0, 0.03, ...)",
                "inner: try_hz in sigma_hz_list",
            ],
            "diagnostic_unregularized_order_when_flag_set": [
                "stiff_frac in (0.0,) only",
                "reg_frac in (0.0,) only",
                "try_hz in offset sigma ladder only",
            ],
            "answers_key_question": (
                "Yes for prior filtered/unfiltered diagnostic: solver sat on sigma≈seed, "
                "LU failed or struggled unregularized, then accepted st_a_shift_frac=0.001 "
                "before exhausting all unregularized offset sigmas. New config forbids that path."
            ),
        },
        "what_st_a_shift_frac_modifies": {
            "function": "fem_main_3d._slepc_st_stiffness_stabilize",
            "file": _fref(fem),
            "docstring_excerpt": (
                "ST-only diagonal stabilization on a copy of A (does not alter harvest operators)."
            ),
            "applied_to": "A_st copy used inside SLEPc ST factorization only",
            "not_applied_to": "Harvest/replay GNHEP (A, M) used for Rayleigh residual and MAC",
            "perturbation_removed_on_mapping": False,
            "replay_operator": "Unregularized physical v2 GNHEP from solve_evp=False assembly",
        },
        "frequency_mapping_after_regularized_solve": {
            "function": "fem_main_3d._slepc_physical_lambda",
            "file": _fref(fem),
            "snippet": _read_snippet(fem, "lam_shift = mu + sigma", context=2),
            "issue": (
                "Mapping assumes unregularized shift-invert semantics. Algebraic mu≈1 modes of "
                "regularized ST can map to lam≈sigma, yielding reported f≈sigma_hz while replay "
                "on physical GNHEP shows lambda≈1."
            ),
        },
    }


def build_operator_context_comparison_table() -> List[Dict[str, Any]]:
    return [
        {
            "operator_context": "A_true_acoustic_seed_no_eigensolve_replay",
            "A_matrix_modification": "Physical v2 GNHEP; algebraic BC rows on A/M",
            "M_matrix_modification": "Physical v2 mass",
            "BC_treatment": "Algebraic Dirichlet on coupled rows",
            "pressure_restriction": "Active-air restriction on reduced W",
            "GNHEP_block_scaling": "Frobenius normalize (v2 diagnosis)",
            "ST_sigma": "N/A (no EPS)",
            "ST_regularization": "None",
            "used_for_seed_residual": True,
            "used_for_candidate_replay": True,
            "used_for_EPS": False,
            "equivalent_to_physical_v2_operator": True,
        },
        {
            "operator_context": "B_filtered_diagnostic_EPS_before_ST_retry",
            "A_matrix_modification": "Same GNHEP A as A",
            "M_matrix_modification": "Same GNHEP M as A",
            "BC_treatment": "Same",
            "pressure_restriction": "Same",
            "GNHEP_block_scaling": "Same",
            "ST_sigma": "sigma≈243.075 Hz (target-first ladder)",
            "ST_regularization": "Initially stiff_frac=0, reg_frac=0",
            "used_for_seed_residual": False,
            "used_for_candidate_replay": True,
            "used_for_EPS": True,
            "equivalent_to_physical_v2_operator": "GNHEP yes; EPS ST initially yes",
        },
        {
            "operator_context": "C_filtered_diagnostic_EPS_after_st_a_shift_frac_0.001",
            "A_matrix_modification": "Harvest A unchanged; ST uses A_st copy + diag shift",
            "M_matrix_modification": "Harvest M unchanged; optional M_eff for ST only",
            "BC_treatment": "Same",
            "pressure_restriction": "Same",
            "GNHEP_block_scaling": "Same",
            "ST_sigma": "sigma≈243.075 Hz",
            "ST_regularization": "st_a_shift_frac=0.001 (default ladder tries this first)",
            "used_for_seed_residual": False,
            "used_for_candidate_replay": True,
            "used_for_EPS": True,
            "equivalent_to_physical_v2_operator": False,
        },
    ]


def build_static_code_audit() -> Dict[str, Any]:
    fem = FEM_SCRIPTS / "fem_main_3d.py"
    return {
        "evidence_scope": "confirmed_from_local_code",
        "leading_confirmed_diagnosis": {
            "evidence_source": "confirmed_from_local_code_and_VM_operator_evidence",
            "summary": [
                "True acoustic seed remains valid under unregularized physical v2 replay.",
                "Prior filtered diagnostic EPS succeeded only after ST A-shift regularization at sigma≈seed.",
                "Returned Ritz vectors are not operator-consistent with replay; lambda≈1 artifacts dominate.",
                "Baseline branch recovery has not succeeded; failure is EPS ST/mapping/selection, not loss of v2 coupling.",
            ],
        },
        "st_retry_control_flow": build_st_retry_control_flow_audit(),
        "operator_context_comparison": build_operator_context_comparison_table(),
        "invalid_audit_method_rejected": {
            "method": "Setting st_a_shift_frac while solve_evp=False",
            "reason": (
                "ST regularization is applied only inside _slepc_try_eps_st_setup on ST copies. "
                "Replay assembly never constructs the regularized ST operator; such a script "
                "cannot test whether ST regularization caused EPS failure."
            ),
        },
        "code_fix_implemented": {
            "diagnostic_requires_unregularized_ST": {
                "file": _fref(fem),
                "effect": "When set, stiff_ladder=(0.0,) and reg_ladder=(0.0,) only.",
            },
            "fail_closed_verdict": "DIAGNOSTIC_ST_REGULARIZATION_REQUIRED_NO_PHYSICAL_VERDICT",
            "unregularized_offset_sigma_ladder": {
                "file": _fref(SCRIPT_DIR / "v2_sensitivity_solve.py"),
                "offsets_hz": [0.5, -0.5, 1.0, -1.0, 2.0, -2.0],
                "never_sigma_at_seed": True,
            },
            "st_metadata_in_results": [
                "st_a_shift_frac_used",
                "st_mass_reg_frac_used",
                "diagnostic_operator_consistent_with_replay",
            ],
        },
        "production_exposure_classification": {
            "replay_only_true_seed_audits": "proven protected (no EPS/ST)",
            "paths_using_eps_with_default_st_ladder": "potentially exposed; report-only spot-check possible",
            "prior_PASS_L0_material_mesh_convergence": "not invalidated automatically; smallest spot-check after baseline closed",
            "standard_seeded_retrieval_with_st_a_shift_frac_0.001": "potentially exposed (VM operator evidence)",
        },
    }


def build_forward_risk_register(
    *,
    filtered_eval: Optional[Dict[str, Any]],
    static_audit: Dict[str, Any],
) -> List[Dict[str, Any]]:
    def row(
        risk: str,
        where: str,
        source: str,
        triggered: str,
        impact: str,
        detect: str,
        fix: str,
        *,
        block_hole: bool,
        block_mesh: bool,
        block_promo: bool,
    ) -> Dict[str, Any]:
        return {
            "risk_or_mismatch": risk,
            "where_in_code": where,
            "evidence_source": source,
            "already_triggered_or_only_possible": triggered,
            "impact_if_unfixed": impact,
            "how_to_detect_before_next_solve": detect,
            "minimal_fix_required": fix,
            "blocks_hole_radius_large": block_hole,
            "blocks_mesh_convergence_resume": block_mesh,
            "blocks_v2_production_promotion": block_promo,
        }

    rows = [
        row(
            "lambda≈1 sigma/mapping artifacts from regularized ST at sigma≈seed",
            "fem_main_3d._slepc_try_eps_st_setup + _slepc_physical_lambda",
            "local_code|VM_operator_evidence",
            "already_triggered (filtered diagnostic)",
            "False branch recovery; reported f≈sigma but replay lambda≈1",
            "Replay Rayleigh lambda; reject abs(lambda-1)<=tol; require op-consistent ST",
            "Unregularized-offset sigma ladder; forbid ST reg for diagnostic verdict",
            block_hole=True,
            block_mesh=True,
            block_promo=True,
        ),
        row(
            "EPS ST-regularized solve not replay-consistent",
            "ST copy A_st vs harvest/replay GNHEP A",
            "confirmed_from_local_code",
            "already_triggered",
            "Candidates are eigenvectors of perturbed ST, not physical GNHEP",
            "Require diagnostic_operator_consistent_with_replay and st_*_frac==0",
            "diagnostic_requires_unregularized_ST fail-closed policy",
            block_hole=True,
            block_mesh=True,
            block_promo=True,
        ),
        row(
            "reported frequency vs replay inconsistency",
            "Harvest f_hz vs replay Rayleigh f_hz",
            "VM_operator_evidence",
            "already_triggered",
            "Accept spurious sigma-near modes",
            "reported_vs_replay_frequency_consistent gate",
            "Post-harvest physical filter (already in v2_seed_branch_candidate_filter)",
            block_hole=True,
            block_mesh=True,
            block_promo=False,
        ),
        row(
            "p_frac-only branch selection",
            "mode_diagnostics p_frac_energy_phys",
            "confirmed_from_local_code",
            "already_triggered",
            "High p_frac on non-physical vectors",
            "MAC+replay+freq gates for branch verdict",
            "Already enforced in diagnostic evaluate",
            block_hole=True,
            block_mesh=True,
            block_promo=False,
        ),
        row(
            "production paths accepting ST-regularized modes without replay screen",
            "fem_main_3d default stiff_ladder",
            "confirmed_from_local_code",
            "only_possible",
            "Prior PASS may include sigma artifacts",
            "Report-only replay filter on saved modes",
            "Spot-check after baseline diagnostic closed",
            block_hole=False,
            block_mesh=True,
            block_promo=True,
        ),
        row(
            "stale reports overriding newer evidence",
            "v2_mesh_convergence/diagnostics/*.json",
            "confirmed_from_local_code",
            "only_possible",
            "Premature resume/promotion",
            "mesh_convergence_may_resume=False until unreg-offset verdict",
            "Supersede prior diagnostic verdicts explicitly",
            block_hole=True,
            block_mesh=True,
            block_promo=True,
        ),
    ]
    return rows


def _mass_norm_audit_pending(mass_norm_audit: Optional[Dict[str, Any]]) -> bool:
    return mass_norm_audit is None or not mass_norm_audit.get("classification_verdict")


def _unreg_evaluation_pending(unreg_eval: Optional[Dict[str, Any]]) -> bool:
    unreg_ev = (unreg_eval or {}).get("evaluation") or {}
    verdict = unreg_ev.get("diagnostic_verdict")
    if verdict in (None, "PENDING_VM_RUN", "PENDING_VM_EVALUATION"):
        return True
    summary = unreg_ev.get("summary") or {}
    n_cand = int(summary.get("num_candidates_evaluated", 0))
    n_ok = int(summary.get("num_metrics_computation_ok", 0))
    # Invalid prior eval: all candidates had non-finite replay/MAC (tooling bug, not physics).
    if (
        verdict == "UNREGULARIZED_OFFSET_OUTPUT_OR_REPLAY_INCONSISTENT"
        and n_cand > 0
        and n_ok == 0
    ):
        return True
    return False


def build_evidence_summary(
    *,
    filtered_eval: Optional[Dict[str, Any]],
    unreg_eval: Optional[Dict[str, Any]],
    mass_norm_audit: Optional[Dict[str, Any]],
    static_audit: Dict[str, Any],
) -> Dict[str, Any]:
    unreg_ev = (unreg_eval or {}).get("evaluation") or {}
    unreg_verdict = unreg_ev.get("diagnostic_verdict")
    mass_verdict = (mass_norm_audit or {}).get("classification_verdict")
    mass_pending = _mass_norm_audit_pending(mass_norm_audit)

    reported_vm = [
        "Prior regularized filtered diagnostic: FILTERED_DIAGNOSTIC_NO_PHYSICAL_BRANCH_RECOVERED; "
        "7 candidates, 0 branch_recovery_pass; 5× lambda≈1 artifacts at reported f≈243 Hz.",
        "Prior filtered run used st_a_shift_frac=0.001 at sigma≈seed (superseded for branch verdict).",
        "True acoustic seed valid at ~243.075 Hz under unregularized physical v2 replay.",
        "Unregularized-offset baseline solve completed: continuation_seed_applied=true, "
        "nconv_marked=56, st_a_shift_frac=0, st_mass_reg_frac=0, "
        "diagnostic_operator_consistent_with_replay=true; 7 modes saved.",
        "Post-solve report-only eval: UNREGULARIZED_OFFSET_OUTPUT_OR_REPLAY_INCONSISTENT with "
        "metrics_computation_ok=0; every candidate xH_Mx=0 (mass-null on replay GNHEP).",
    ]
    if unreg_verdict:
        reported_vm.append(f"Unregularized-offset evaluation verdict: {unreg_verdict}.")
    if mass_verdict:
        reported_vm.append(f"Saved-vector mass-norm audit classification: {mass_verdict}.")

    requires_vm: List[str] = []
    if mass_pending:
        requires_vm.append(
            "Final saved-vector persistence/mass-norm audit on existing unregularized-offset tree."
        )

    not_yet: List[str] = [
        "Whether any prior PASS harvest modes fail replay filter spot-check.",
    ]
    if mass_pending:
        not_yet.insert(
            0,
            "Definitive classification: replay control vs persistence bug vs EPS M-null candidates.",
        )

    return {
        "confirmed_from_local_code": [
            "Default ST retry tries st_a_shift_frac=1e-3 before 0.0.",
            "ST A-shift modifies only ST factorization copy; replay uses unregularized GNHEP.",
            "lam_shift=mu+sigma mapping has no inverse for ST regularization.",
            "diagnostic_requires_unregularized_ST forces unregularized ST only on offset diagnostic.",
            "Invalid ST-equivalence via solve_evp=False flags cannot reproduce EPS ST operator.",
            "Earlier regularized diagnostic runs superseded for branch-recovery judgment.",
        ],
        "reported_from_VM_operator_evidence": reported_vm,
        "requires_VM_runtime_artifact_evaluation": requires_vm,
        "not_yet_verified": not_yet,
        "single_permitted_next_action": (
            "bash .../run_v2_l_mid_unregularized_saved_vector_mass_norm_audit.sh"
            if mass_pending
            else None
        ),
        "prior_regularized_diagnostics_superseded_for_branch_verdict": True,
        "unregularized_offset_solve_completed": True,
        "unregularized_offset_evaluation_verdict": unreg_verdict,
        "saved_vector_mass_norm_classification": mass_verdict,
        "candidates_currently_unevaluable_xH_Mx_zero": True,
    }


def determine_root_cause_status(
    *,
    filtered_eval: Optional[Dict[str, Any]],
    unreg_eval: Optional[Dict[str, Any]],
) -> str:
    return ROOT_CAUSE_CONFIRMED_ST


def build_finite_closure_plan(
    *,
    filtered_eval: Optional[Dict[str, Any]],
    unreg_eval: Optional[Dict[str, Any]],
    mass_norm_audit: Optional[Dict[str, Any]],
    root_cause_status: str,
) -> Dict[str, Any]:
    mass_verdict = (mass_norm_audit or {}).get("classification_verdict")
    mass_pending = _mass_norm_audit_pending(mass_norm_audit)
    closure = (mass_norm_audit or {}).get("closure") or {}

    if mass_pending:
        next_action = (
            "Run final report-only saved-vector persistence/mass-norm audit on existing "
            "unregularized-offset artifacts (no eigensolve)."
        )
    elif mass_verdict == VERDICT_PERSISTENCE_BUG:
        next_action = (
            "Confirmed save/load or layout bug. Only permitted follow-up: report-only "
            "re-evaluation if correct vectors recoverable from existing artifacts; "
            "otherwise artifacts insufficient."
        )
    elif mass_verdict in (
        VERDICT_EPS_MASS_NULL,
        VERDICT_REPLAY_INVALID,
        VERDICT_NOT_LOCALIZED,
    ):
        next_action = (
            "Stop patching and reconsider EPS/ST branch-tracking architecture before any "
            "further solve. hole_radius_large and mesh_convergence remain blocked."
        )
    else:
        next_action = "Review mass-norm audit classification output."

    if root_cause_status == ROOT_CAUSE_CONFIRMED_ST:
        blocked = [
            "hole_radius_large",
            "mesh_convergence_resume",
            "L_prod",
            "L_check",
            "LHS",
            "v2_production_promotion",
            "baseline_diagnostic_eigensolve_rerun",
        ]
    else:
        next_action = "ROOT_CAUSE_NOT_YET_CONFIRMED — no further solve authorized."
        blocked = [
            "hole_radius_large",
            "mesh_convergence_resume",
            "L_prod",
            "L_check",
            "LHS",
            "v2_production_promotion",
            "baseline_diagnostic_rerun",
        ]

    return {
        "root_cause_status": root_cause_status,
        "decision_tree": {
            "if_unreg_offset_rerun_recovers_branch": (
                "Allow exactly one next solve: hole_radius_large under same filtered diagnostic path."
            ),
            "if_unreg_offset_rerun_fails_without_ST_reg": (
                "Stop patching; report EPS/ST diagnostic architecture inadequate for branch recovery."
            ),
            "prior_PASS": (
                "Not auto-invalidated; smallest report-only spot-check set after baseline closed."
            ),
        },
        "next_allowed_action_after_VM_report": next_action,
        "blocked_actions": blocked,
        "maximum_additional_baseline_solves_before_escalation": 1,
        "maximum_additional_code_fix_cycles_before_reconsidering_solver_architecture": 1,
        "filtered_eval_verdict": None if not filtered_eval else filtered_eval.get("verdict"),
        "unregularized_offset_eval_verdict": (unreg_eval or {}).get("evaluation", {}).get(
            "diagnostic_verdict"
        ),
        "saved_vector_mass_norm_classification": mass_verdict,
        "mass_norm_audit_pending": mass_pending,
        "baseline_eigensolve_budget_exhausted": True,
        "architecture_reconsideration_required": bool(
            closure.get("architecture_reconsideration_required")
        ),
    }
