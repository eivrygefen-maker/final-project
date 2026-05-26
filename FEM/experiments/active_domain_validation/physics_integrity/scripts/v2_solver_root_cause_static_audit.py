#!/usr/bin/env python3
"""
Local (workspace-only) static audit for seed-branch recovery diagnostics.

This module MUST NOT read VM-only runtime artifacts. It produces a code-derived
root-cause audit and a forward risk register; the VM runtime merge happens in
run_v2_solver_root_cause_and_forward_risk_audit.py.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
FEM_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"

import sys

if str(FEM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FEM_SCRIPTS))


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
    return f"(needle {needle!r} not found in {path})"


def _load_v2_config() -> Dict[str, Any]:
    cfg_path = FEM_SCRIPTS.parent / "experiments" / "active_domain_validation" / "physics_integrity" / "configs" / "coupled_physical_core_v2.json"
    if not cfg_path.is_file():
        # fallback to checked-in path used elsewhere
        cfg_path = FEM_SCRIPTS.parent / "configs" / "coupled_physical_core_v2.json"
    if not cfg_path.is_file():
        return {}
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


VERDICT_FILTERED_BRANCH_RECOVERED = "FILTERED_DIAGNOSTIC_BRANCH_RECOVERED"
VERDICT_FILTERED_NO_BRANCH = "FILTERED_DIAGNOSTIC_NO_PHYSICAL_BRANCH_RECOVERED"
VERDICT_FILTERED_INCONSISTENT = "FILTERED_DIAGNOSTIC_OUTPUT_OR_REPLAY_INCONSISTENT"


def build_static_code_audit() -> Dict[str, Any]:
    """Return only locally checkable facts (confirmed_from_local_code)."""
    fem_main = FEM_SCRIPTS / "fem_main_3d.py"

    # v2 scripts live in the physics_integrity experiment scripts dir.
    v2_dir = SCRIPT_DIR
    v2_sensitivity_solve = v2_dir / "v2_sensitivity_solve.py"
    run_diag = v2_dir / "run_v2_l_mid_seed_branch_recovery_diagnostic.py"
    filter_mod = v2_dir / "v2_seed_branch_candidate_filter.py"
    filtered_eval = v2_dir / "run_v2_l_mid_seed_branch_filtered_evaluation.py"
    root_report = v2_dir / "run_v2_l_mid_seed_branch_vm_consolidated_report.sh"

    config = _load_v2_config()
    prod_eps_reject_sigma_spurious = (
        (config.get("solver") or {}).get("eps_reject_sigma_spurious")
        if isinstance(config, dict)
        else None
    )

    return {
        "evidence_scope": "confirmed_from_local_code",
        "files": {
            "fem_main_3d": _fref(fem_main),
            "v2_sensitivity_solve": _fref(v2_sensitivity_solve),
            "run_diag": _fref(run_diag),
            "filter_mod": _fref(filter_mod),
            "filtered_eval": _fref(filtered_eval),
            "consolidated_report_sh": _fref(root_report),
        },
        "code_path_reviewed": {
            "eps_st_lambda_mapping": {
                "where_in_code": "fem_main_3d._slepc_physical_lambda",
                "snippet": _read_snippet(fem_main, "lam_shift = mu + sigma", context=2),
                "confirmed_behavior": (
                    "Shift path can map SLEPc Ritz mu≈1 into physical lambda≈sigma, so "
                    "reported frequency can align with sigma even when the vector is an "
                    "algebraic/identity-like mode."
                ),
            },
            "algebraic_dirichlet_rows": {
                "where_in_code": "fem_main_3d._coupled_algebraic_dirichlet_rows + _petsc_mat_zero_dirichlet_rows",
                "snippet_dirichlet": _read_snippet(
                    fem_main, "def _coupled_algebraic_dirichlet_rows", context=3
                ),
                "confirmed_behavior": (
                    "Coupled Dirichlet/BC dofs are enforced via algebraic row operations on "
                    "assembled A/M, producing identity-like algebraic rows that can yield "
                    "sigma-spurious candidates unless filtered with replay-based gates."
                ),
            },
            "production_harvest_sigma_rejection": {
                "where_in_code": "fem_main_3d EPS harvest loop rejects eps_reject_sigma_spurious",
                "snippet": _read_snippet(fem_main, "if reject_sigma_spurious and p_frac < sigma_p_frac_max", context=2),
                "confirmed_default": {
                    "prod_cfg_eps_reject_sigma_spurious": prod_eps_reject_sigma_spurious,
                },
                "confirmed_behavior": (
                    "When enabled, near-ST sigma modes with tiny pressure participation (p_frac) "
                    "can be rejected at harvest time. Diagnostic modes may disable this."
                ),
            },
            "diagnostic_cfg_deltas": {
                "where_in_code": "v2_sensitivity_solve._apply_seed_branch_recovery_diagnostic_solver_cfg and filtered diagnostic cfg",
                "snippet_unfiltered": _read_snippet(
                    v2_sensitivity_solve,
                    "eps_reject_sigma_spurious = False",
                    context=3,
                ),
                "snippet_filtered": _read_snippet(
                    v2_sensitivity_solve,
                    "--seed-branch-filtered-diagnostic",
                    context=1,
                ),
                "confirmed_behavior": (
                    "Unfiltered diagnostic disables harvest-time sigma spurious rejection and "
                    "saves all candidates for inspection. Filtered diagnostic rerun enables "
                    "harvest-time sigma spurious rejection, while still relying on post-save "
                    "replay/MAC filtering for final eligibility."
                ),
            },
            "candidate_selection_and_replay": {
                "where_in_code": "run_v2_l_mid_seed_branch_recovery_diagnostic._evaluate + v2_seed_branch_candidate_filter",
                "snippet_diag_evaluate": _read_snippet(
                    run_diag, "replay_rayleigh_eigenvalue", context=3
                ),
                "snippet_filter_mod": _read_snippet(
                    filter_mod, "assess_physical_eligibility", context=2
                ),
                "confirmed_behavior": (
                    "Final physical eligibility is replay-based: it requires finite Rayleigh "
                    "values, reported-vs-replay frequency consistency, residual screening, "
                    "MAC gate to the true seed ordering, and explicit rejection of lambda≈1."
                ),
            },
            "mode_save_indexing": {
                "where_in_code": "v2_sensitivity_solve mode save loop mode_{hz_tag}_{j:03d}.smx.npz; frequency_hz stored per j",
                "snippet": _read_snippet(
                    v2_sensitivity_solve,
                    "mode_{hz_tag}_{j:03d}",
                    context=2,
                ),
                "confirmed_risk": (
                    "If an evaluator mismatches a saved vector with its stored frequency/row "
                    "metadata (vector_file vs mode_index), replay can expose inconsistency."
                ),
            },
        },
        "scipy_numpy_usage": {
            "confirmed_local_only_note": (
                "Some mode diagnostics / save utilities use SciPy sparse for mode I/O. "
                "The replay/MAC filter path used by the report-only evaluators is NumPy/PETSc "
                "centric, but the VM-side mode load stack may still show SciPy/NumPy compatibility warnings."
            ),
            "where_in_code_scipy_imported": [
                _fref(FEM_SCRIPTS / "fem_mode_array_utils.py"),
                _fref(SCRIPT_DIR / "mode_diagnostics.py"),
            ],
        },
    }


def build_forward_risk_register(
    *,
    filtered_eval: Dict[str, Any] | None,
    static_audit: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Return risk rows with evidence_source and blocks flags."""

    verdict = None if not filtered_eval else filtered_eval.get("verdict")
    any_branch_recovered = verdict == VERDICT_FILTERED_BRANCH_RECOVERED

    # Conservative: until VM evidence is merged, blocks are True.
    blocks_hole = not any_branch_recovered

    # Helper
    def _row(
        risk_or_mismatch: str,
        where_in_code: str,
        evidence_source: str,
        already_triggered_or_only_possible: str,
        impact_if_unfixed: str,
        how_to_detect_before_next_solve: str,
        minimal_fix_required: str,
        *,
        blocks_hole_radius_large: bool,
        blocks_mesh_convergence_resume: bool,
        blocks_v2_production_promotion: bool,
    ) -> Dict[str, Any]:
        return {
            "risk_or_mismatch": risk_or_mismatch,
            "where_in_code": where_in_code,
            "evidence_source": evidence_source,
            "already_triggered_or_only_possible": already_triggered_or_only_possible,
            "impact_if_unfixed": impact_if_unfixed,
            "how_to_detect_before_next_solve": how_to_detect_before_next_solve,
            "minimal_fix_required": minimal_fix_required,
            "blocks_hole_radius_large": blocks_hole_radius_large,
            "blocks_mesh_convergence_resume": blocks_mesh_convergence_resume,
            "blocks_v2_production_promotion": blocks_v2_production_promotion,
        }

    rows = [
        _row(
            "lambda≈1 sigma/mapping artifacts",
            "fem_main_3d._slepc_physical_lambda + diagnostic cfg eps_reject_sigma_spurious=False",
            "local_code",
            "already_triggered (VM evidence on unfiltered mode)",
            "False branch recovery near seed frequency",
            "Recompute replay Rayleigh lambda; reject abs(lambda-1)<=tol; require freq consistency + residual gate",
            "Keep harvest-time reject off for exploration only, but require post-save replay/MAC filtering; keep lambda≈1 explicit reject",
            blocks_hole_radius_large=blocks_hole,
            blocks_mesh_convergence_resume=True,
            blocks_v2_production_promotion=True,
        ),
        _row(
            "reported frequency vs replay inconsistency",
            "run_v2_l_mid_seed_branch_recovery_diagnostic._evaluate and filtered evaluation script",
            "local_code",
            "already_triggered (VM evidence on focal candidate)",
            "Reported-vs-replay mismatch hides algebraic identity vectors as physical candidates",
            "Gate with reported_vs_replay_frequency_consistent + residual screening in report-only evaluators",
            "Make replay-based gating the only path to branch verdicts",
            blocks_hole_radius_large=True,
            blocks_mesh_convergence_resume=True,
            blocks_v2_production_promotion=False,
        ),
        _row(
            "p_frac-only branch selection risk",
            "mode_diagnostics.compute_mass_energy_participation + harvest ranking knobs",
            "local_code",
            "only_possible (but common)",
            "High p_frac on non-physical vectors yields false positives",
            "Require MAC + replay residual + freq consistency; never accept p_frac without replay",
            "Ensure branch verdict uses pressure MAC to seed ordering + replay residual",
            blocks_hole_radius_large=True,
            blocks_mesh_convergence_resume=True,
            blocks_v2_production_promotion=False,
        ),
        _row(
            "mode-file/result-row index mismatch",
            "v2_sensitivity_solve mode save loop (j) vs evaluator vector selection",
            "local_code",
            "only_possible",
            "MAC/replay computed on a different vector than the claimed metadata",
            "Enumerate vector_file, mode_index, reported f together in report-only tables; verify same saved vector is replayed",
            "Use mode_index+vector_path join and validate counts in VM evaluator",
            blocks_hole_radius_large=False,
            blocks_mesh_convergence_resume=True,
            blocks_v2_production_promotion=False,
        ),
        _row(
            "stale reports overriding newer evidence",
            "v2_mesh_convergence/diagnostics JSON verdict fields and superseded markers",
            "local_code",
            "only_possible",
            "Old unfiltered verdict is used to resume mesh convergence",
            "Hard-disable resume flags until consolidated VM bundle verdict is merged",
            "Update superseded_* markers and keep mesh_convergence_may_resume=False until filtered_evaluation completes",
            blocks_hole_radius_large=True,
            blocks_mesh_convergence_resume=True,
            blocks_v2_production_promotion=True,
        ),
        _row(
            "normal production path possible exposure",
            "coupled_physical_core_v2.json eps_reject_sigma_spurious and production harvest ranking",
            "requires_VM_evaluation",
            "only_possible",
            "Production harvest might also admit near-sigma spurious modes",
            "Verify production solve artifacts with the same replay-based filtered evaluator (no eigensolve) on VM",
            "Use the filtered evaluator as a post-solve gate on production artifacts before promoting",
            blocks_hole_radius_large=False,
            blocks_mesh_convergence_resume=True,
            blocks_v2_production_promotion=True,
        ),
        _row(
            "SciPy/NumPy compatibility warning",
            "fem_mode_array_utils / mode_diagnostics sparse mode I/O",
            "requires_VM_evaluation",
            "only_possible",
            "Sparse mode load or p_frac diagnostics could be corrupted",
            "On VM, check import warnings and compare replay/MAC consistency for the same saved vector",
            "If warnings exist, ensure filtering replay uses NumPy vectors loaded reliably; consider switching I/O path to dense if needed",
            blocks_hole_radius_large=False,
            blocks_mesh_convergence_resume=False,
            blocks_v2_production_promotion=False,
        ),
        _row(
            "filtering applied only after solve vs before save",
            "v2_sensitivity_solve seeded diagnostic saves all candidates; report-only evaluator filters afterwards",
            "local_code",
            "already_triggered (7 saved modes on VM)",
            "Disk artifacts include spurious modes; wrong selection logic could still pick them",
            "Branch verdict must be computed only from filtered evaluator result; mark unfiltered verdicts as superseded",
            "Use filtered evaluator output JSON as the sole verdict source for gating next actions",
            blocks_hole_radius_large=True,
            blocks_mesh_convergence_resume=True,
            blocks_v2_production_promotion=False,
        ),
        _row(
            "resume logic reusing stale artifacts",
            "mesh_convergence flags and diagnostic verdict fields consumed by later scripts",
            "local_code",
            "only_possible",
            "Resume mesh convergence without passing filtered evidence",
            "Ensure later stages check that filtered_evaluation JSON exists and runtime_evaluation_completed=True",
            "Add explicit gating checks in the orchestration code (future work)",
            blocks_hole_radius_large=True,
            blocks_mesh_convergence_resume=True,
            blocks_v2_production_promotion=True,
        ),
    ]

    # If VM already reports recovered branch, relax the hole_radius_large block for the lambda≈1 row.
    if any_branch_recovered:
        for r in rows:
            if r["risk_or_mismatch"] == "lambda≈1 sigma/mapping artifacts":
                r["blocks_hole_radius_large"] = False
    return rows


def build_evidence_summary(
    *,
    filtered_eval: Dict[str, Any] | None,
    static_audit: Dict[str, Any],
) -> Dict[str, Any]:
    """Combine confirmed local code facts with VM-provided evidence (pending)."""
    return {
        "confirmed_from_local_code": [
            "Shift/invert lambda mapping exists in fem_main_3d._slepc_physical_lambda (lam_shift = mu + sigma).",
            "Coupled algebraic Dirichlet row operations exist in fem_main_3d for BC enforcement; identity-like rows can create sigma artifacts.",
            "Diagnostic solver cfg disables eps_reject_sigma_spurious for unfiltered diagnostic, and filtered diagnostic enables it (experiment-only rerun).",
            "Report-only filtered evaluators compute eligibility via replay Rayleigh lambda/frequency, replay residual, reported-vs-replay consistency, explicit lambda≈1 rejection, and MAC to validated seed ordering.",
            "Mode saving uses mode_index=j and vector_file mode_{hz_tag}_{j:03d}.smx.npz; evaluators must match metadata to vector_file.",
        ],
        "reported_from_VM_operator_evidence": [
            "Unfiltered baseline diagnostic selected an algebraic lambda≈1 sigma/mapping artifact as recovered near 243.075 Hz (as provided by operator evidence).",
            "Filtered diagnostic run saved 7 candidates with report-only physical filtering; f_branch is NaN as expected when saving all modes (operator evidence).",
        ],
        "requires_VM_runtime_artifact_evaluation": [
            "Whether any of the 7 filtered candidates passes branch_recovery_pass gates (requires running run_v2_l_mid_seed_branch_filtered_evaluation.py on VM).",
        ],
        "not_yet_verified": [
            "The filtered_evaluation verdict itself (must not be claimed until VM script is executed).",
            "Whether production L_mid harvest artifacts admit spurious lambda≈1 candidates; requires production-side VM evaluator usage.",
        ],
        "single_vm_run_combines": [
            "Run filtered candidate evaluation on VM (v2_l_mid_seed_branch_recovery_filtered_evaluation.{json,md}).",
            "Run consolidated root-cause + forward risk audit merging that JSON (v2_solver_root_cause_and_forward_risk_audit.{json,md}).",
        ],
        "runtime_evaluation_completed": bool(filtered_eval and filtered_eval.get("runtime_evaluation_completed", False)),
        "filtered_evaluation_verdict": None if not filtered_eval else filtered_eval.get("verdict"),
    }


def build_finite_closure_plan(*, filtered_eval: Dict[str, Any] | None) -> Dict[str, Any]:
    fe_verdict = None if not filtered_eval else filtered_eval.get("verdict")
    pending = True if not filtered_eval else filtered_eval.get("verdict_pending_until_vm_run", True)

    if pending or fe_verdict is None:
        next_allowed_action = "Run consolidated VM report-only bundle; await v2_l_mid_seed_branch_recovery_filtered_evaluation verdict."
        blocked = [
            "hole_radius_large",
            "mesh_convergence_resume",
            "L_prod",
            "L_check",
            "LHS",
            "v2_production_promotion",
        ]
    elif fe_verdict == VERDICT_FILTERED_BRANCH_RECOVERED:
        next_allowed_action = (
            "Allow exactly one next solve: hole_radius_large under the same filtered diagnostic path "
            "(after operator review of filtered_evaluation JSON)."
        )
        blocked = [
            "mesh_convergence_resume",
            "L_prod",
            "L_check",
            "LHS",
            "v2_production_promotion",
            "unfiltered_baseline_diagnostic_rerun",
        ]
    elif fe_verdict == VERDICT_FILTERED_NO_BRANCH:
        next_allowed_action = (
            "Identify concrete code-level root cause from merged static+VM evidence; "
            "allow at most one further baseline rerun using the filtered diagnostic path."
        )
        blocked = [
            "hole_radius_large",
            "mesh_convergence_resume",
            "L_prod",
            "L_check",
            "LHS",
            "v2_production_promotion",
        ]
    else:
        next_allowed_action = (
            "Resolve output/replay inconsistency: re-run filtered evaluation on VM (no eigensolve) before any solve."
        )
        blocked = [
            "hole_radius_large",
            "mesh_convergence_resume",
            "L_prod",
            "L_check",
            "LHS",
            "v2_production_promotion",
        ]

    return {
        "next_allowed_action_after_VM_report": next_allowed_action,
        "blocked_actions": blocked,
        "maximum_additional_baseline_solves_before_escalation": 1,
        "maximum_additional_code_fix_cycles_before_reconsidering_solver_architecture": 1,
        "decision_tree": {
            "if_filtered_candidate_report_finds_valid_recovered_baseline_branch": (
                "Allow exactly one next solve: hole_radius_large under the same filtered diagnostic path."
            ),
            "if_no_valid_recovered_baseline_branch": (
                "Identify concrete code-level root cause; at most one further baseline filtered rerun before escalation."
            ),
            "if_exposure_diagnostic_only": (
                "Document conclusion after VM evidence merge; continue baseline → hole_radius_large → mesh-convergence decision."
            ),
        },
        "vm_report_pending": pending,
        "filtered_verdict": fe_verdict,
    }

