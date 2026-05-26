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


def main() -> int:
    preflight: Optional[Dict[str, Any]] = None
    if PREFLIGHT_JSON.is_file():
        preflight = json.loads(PREFLIGHT_JSON.read_text(encoding="utf-8"))
    mapping_inv = (
        json.loads(MAPPING_JSON.read_text(encoding="utf-8")) if MAPPING_JSON.is_file() else {}
    )

    applicability = (preflight or {}).get("PGNHEP_purification_applicability", "unresolved")
    stage1_authorized = False
    stage2_only = False
    if applicability == "supported_for_stage1_test_pending_vm_confirmation":
        stage1_authorized = False  # still blocked until human review
    elif applicability == "not_justified_use_nullspace_reduction_plan":
        stage2_only = True

    report: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "strategy": "finite_solver_rehabilitation_two_stages",
        "confirmed_vm_evidence": {
            "seed_xH_Mx_finite_nonzero": True,
            "unregularized_offset_solve_operator_consistent": True,
            "all_saved_candidates_xH_Mx_zero": True,
            "classification": "EPS_RETURNED_ONLY_MASS_NULL_CANDIDATES_IN_UNREGULARIZED_SOLVE",
        },
        "stage_0_mapping_fix": MAPPING_FIX_SUMMARY,
        "stage_1": {
            "description": "SLEPc-native singular-mass handling on unregularized ST",
            "blocked_until": [
                "mapping_impact_inventory reviewed",
                "existing_pass_replay_recertification reviewed",
                "v2_st_singular_mass_preflight reviewed",
            ],
            "authorized": stage1_authorized,
            "prepared_command_after_review": (
                "bash FEM/experiments/active_domain_validation/physics_integrity/scripts/"
                "run_v2_l_mid_st_purified_baseline_diagnostic.sh"
                if not stage2_only
                else None
            ),
            "acceptance_gates": [
                "continuation_seed_applied=True",
                "diagnostic_operator_consistent_with_replay=True",
                "actual_st_a_shift_frac=0",
                "actual_st_mass_reg_frac=0",
                "candidate xH_Mx finite and nonzero",
                "reported vs replay frequency consistent",
                "replay residual within tolerance",
                "frequency within 1% of seed",
                "pressure MAC >= 0.85",
            ],
            "production_policy_unchanged": True,
        },
        "stage_2": {
            "description": "Explicit physical null-space reduction (design only unless Stage 1 ruled out)",
            "implement_in_this_step": stage2_only,
            "plan_outline": [
                "Identify mass-null subspace from pressure restriction / algebraic constraints",
                "Build physical pencil on complement of null space",
                "Map seed and saved modes between W and reduced basis",
                "Preserve three-worker overlapping-frequency architecture",
                "Re-evaluate save/load/replay/MAC without changing production defaults",
            ],
            "trigger": "PGNHEP/purification not justified OR single Stage-1 test fails after mapping fix",
        },
        "fallback_if_both_stages_fail": {
            "leading_candidate": "JD/GD on physically cleaned pencil",
            "constraints": [
                "preserve three-worker architecture",
                "benchmark wall time vs valid ST baseline",
                "accept only if physical modes and <=50% runtime increase",
            ],
            "not_first_choice": ["CISS", "new formulation without justification"],
        },
        "preflight_summary": preflight,
        "mapping_inventory_summary": {
            "prior_PASS_auto_invalidated": mapping_inv.get("prior_PASS_auto_invalidated", False),
        },
        "mesh_convergence_may_resume": False,
        "additional_baseline_eigensolve": "blocked_pending_preflight_review",
        "hole_radius_large": "blocked",
    }
    write_json(OUT_JSON, report)

    lines = [
        "# ST singular-mass rehabilitation plan",
        "",
        f"Generated: {report['generated_utc']}",
        "",
        "## Stage 0 (implemented): eigenvalue mapping",
        "",
        f"- {MAPPING_FIX_SUMMARY['new_behavior']}",
        "",
        "## Stage 1",
        "",
        f"- **Authorized:** `{stage1_authorized}` (blocked until report review)",
        f"- **PGNHEP applicability (preflight):** `{applicability}`",
        "",
        "## Stage 2 trigger",
        "",
        "Move to Stage 2 null-space reduction design if PGNHEP/purification is not justified "
        "or the single permitted Stage-1 baseline test fails after mapping fix.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[rehab_plan] wrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
