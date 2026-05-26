#!/usr/bin/env python3
"""Write ST singular-mass rehabilitation plan from code + optional preflight JSON (no solve)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (SCRIPT_DIR,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from v2_eps_mapping_audit_lib import MAPPING_FIX_SUMMARY
from v2_mesh_convergence_common import CONV_DIAG, write_json

OUT_JSON = CONV_DIAG / "v2_st_singular_mass_rehabilitation_plan.json"
OUT_MD = CONV_DIAG / "v2_st_singular_mass_rehabilitation_plan.md"
PREFLIGHT_JSON = CONV_DIAG / "v2_st_singular_mass_preflight.json"
MAPPING_JSON = CONV_DIAG / "v2_eps_mapping_impact_inventory.json"
MAPPING_FIXED_DIAG_JSON = (
    CONV_DIAG / "v2_l_mid_mapping_fixed_unregularized_baseline_diagnostic.json"
)
VM_BASELINE_SHELL = (
    "bash FEM/experiments/active_domain_validation/physics_integrity/scripts/"
    "run_v2_l_mid_mapping_fixed_unregularized_baseline_diagnostic.sh"
)


def main() -> int:
    preflight: Optional[Dict[str, Any]] = None
    if PREFLIGHT_JSON.is_file():
        preflight = json.loads(PREFLIGHT_JSON.read_text(encoding="utf-8"))
    mapping_inv = (
        json.loads(MAPPING_JSON.read_text(encoding="utf-8")) if MAPPING_JSON.is_file() else {}
    )
    mapping_fixed = (
        json.loads(MAPPING_FIXED_DIAG_JSON.read_text(encoding="utf-8"))
        if MAPPING_FIXED_DIAG_JSON.is_file()
        else {}
    )

    applicability = (preflight or {}).get("PGNHEP_purification_applicability")
    if applicability is None:
        applicability = "not_justified_use_nullspace_reduction_plan"
    pgnhep_ruled_out = applicability == "not_justified_use_nullspace_reduction_plan"
    mapping_fixed_ev = (mapping_fixed or {}).get("evaluation") or {}
    mapping_fixed_verdict = mapping_fixed_ev.get("diagnostic_verdict")
    baseline_pending = mapping_fixed_verdict in (
        None,
        "PENDING_VM_SOLVE_AND_EVALUATION",
    )

    report: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "strategy": "finite_solver_rehabilitation_mapping_corrected_baseline_then_stage2",
        "next_allowed_action": (
            "one mapping-corrected unregularized baseline ST diagnostic"
            if baseline_pending
            else "review_mapping_corrected_baseline_verdict"
        ),
        "recommended_vm_command": VM_BASELINE_SHELL if baseline_pending else None,
        "PGNHEP_purification": "ruled_out_in_current_VM_environment",
        "purification": "ruled_out_in_current_VM_environment",
        "stage_2": {
            "description": "Explicit physical null-space reduction",
            "mandatory_only_if": (
                "mapping-corrected unregularized baseline diagnostic fails to recover "
                "physical branch"
            ),
            "authorized_now": False,
            "blocked_until": [
                "mapping_corrected_baseline_diagnostic_completed_and_reviewed",
            ],
            "plan_outline": [
                "Identify mass-null subspace from pressure restriction / algebraic constraints",
                "Build physical pencil on complement of null space",
                "Map seed and saved modes between W and reduced basis",
                "Preserve three-worker overlapping-frequency architecture",
                "Re-evaluate save/load/replay/MAC without changing production defaults",
            ],
        },
        "not_authorized": [
            "PGNHEP/purification in current VM environment",
            "another sigma adjustment",
            "another filter-only EPS rerun",
            "another ST mapping variant",
            "immediate Stage-2 before mapping-corrected baseline completes",
        ],
        "confirmed_vm_evidence": {
            "seed_xH_Mx_finite_nonzero": True,
            "pre_mapping_fix_unregularized_offset_solve_not_valid_mapping_test": True,
            "seven_saved_candidates_mass_null_not_evidence_against_corrected_mapping": True,
            "PGNHEP_purification_applicability": applicability,
            "has_EPS_ProblemType_PGNHEP": (preflight or {}).get("has_EPS_ProblemType_PGNHEP"),
            "can_set_PGNHEP_without_solve": (preflight or {}).get("can_set_PGNHEP_without_solve"),
            "can_set_purify_without_solve": (preflight or {}).get("can_set_purify_without_solve"),
            "some_modes_valid_physics_wrong_frequency_labels_only": True,
        },
        "stage_0_mapping_fix": MAPPING_FIX_SUMMARY,
        "mapping_corrected_baseline_diagnostic": {
            "authorized": baseline_pending,
            "preserve_all_nconv_candidates": True,
            "physical_eligibility_after_save": True,
            "verdicts": [
                "MAPPING_FIXED_UNREGULARIZED_BASELINE_BRANCH_RECOVERED",
                "MAPPING_FIXED_UNREGULARIZED_BASELINE_NO_PHYSICAL_BRANCH_RECOVERED",
                "MAPPING_FIXED_UNREGULARIZED_BASELINE_OUTPUT_OR_REPLAY_INCONSISTENT",
            ],
            "acceptance_gates": [
                "continuation_seed_applied=True",
                "eps_eigenvalue_semantics=slepc_backtransformed",
                "legacy_double_shift_mapping_disabled=True",
                "diagnostic_operator_consistent_with_replay=True",
                "actual_st_a_shift_frac=0",
                "actual_st_mass_reg_frac=0",
                "candidate xH_Mx finite and nonzero",
                "reported vs replay frequency consistent",
                "replay residual within tolerance",
                "frequency within 1% of seed",
                "pressure MAC >= 0.85",
            ],
            "report_json": str(MAPPING_FIXED_DIAG_JSON),
            "current_verdict": mapping_fixed_verdict,
        },
        "prior_pass_handling": {
            "mesh_topology_gates_preserved": True,
            "true_seed_replay_findings_preserved": True,
            "eps_frequency_labels_pending_recertification": True,
            "prior_PASS_auto_invalidated": mapping_inv.get("prior_PASS_auto_invalidated", False),
        },
        "preflight_summary": preflight,
        "mapping_inventory_summary": {
            "prior_PASS_auto_invalidated": mapping_inv.get("prior_PASS_auto_invalidated", False),
        },
        "mesh_convergence_may_resume": False,
        "additional_baseline_eigensolve": (
            "one_mapping_corrected_unregularized_baseline_authorized"
            if baseline_pending
            else "blocked_pending_baseline_review"
        ),
        "hole_radius_large": "blocked",
        "production_policy_unchanged": True,
    }
    write_json(OUT_JSON, report)

    lines = [
        "# ST singular-mass rehabilitation plan",
        "",
        f"Generated: {report['generated_utc']}",
        "",
        "## Next allowed action",
        "",
        f"- **next_allowed_action:** `{report['next_allowed_action']}`",
        f"- **VM command:** `{report.get('recommended_vm_command')}`",
        "",
        "## PGNHEP / purification",
        "",
        f"- **Status:** `{report['PGNHEP_purification']}`",
        "",
        "## Stage 0 (implemented): eigenvalue mapping",
        "",
        f"- {MAPPING_FIX_SUMMARY['new_behavior']}",
        "",
        "## Mapping-corrected baseline (authorized once)",
        "",
        f"- **Authorized:** `{report['mapping_corrected_baseline_diagnostic']['authorized']}`",
        f"- **Current verdict:** `{mapping_fixed_verdict}`",
        "",
        "## Stage 2",
        "",
        "Mandatory **only if** the mapping-corrected unregularized baseline diagnostic fails "
        "to recover a physical branch. Not authorized before that run completes.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[rehab_plan] wrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
