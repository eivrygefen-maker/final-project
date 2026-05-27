#!/usr/bin/env python3
"""Clean adjudication lane constants and planned policy (no EPS)."""
from __future__ import annotations

from typing import Any, Dict, List

# Isolated output tree for future single EPS adjudication run (not yet authorized).
OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1 = (
    "seed_branch_recovery_diagnostic_mapping_fixed_unregularized_lossless_adjudication_v1"
)
OUT_SUBDIR_LOSSLESS_NULLSPACE_PROJECTED_ADJUDICATION_V1 = (
    "seed_branch_recovery_diagnostic_mapping_fixed_unregularized_lossless_nullspace_projected_adjudication_v1"
)

CASE_ID = "baseline_coupled_v2"
SEED_F_HZ = 243.0754171175576
APPROVED_SIGMA_HZ = 243.5754171175576

AUTHORIZED_EVIDENCE_LAYER_DIFFERENCES: List[str] = [
    "new isolated output directory",
    "lossless candidate persistence before sparsification (.smx.dense.npy authoritative)",
    "complete policy metadata persistence including st_type",
    "evaluator reads lossless vectors as authoritative replay source",
    "legacy sparse .smx.npz optional comparison only",
    "pre-replay filters cannot prevent capture; applied post-replay for attribution only",
]


def planned_clean_lane_policy() -> Dict[str, Any]:
    """Target policy for lossless adjudication v1 (solver/physics identical to approved replacement)."""
    return {
        "case_id": CASE_ID,
        "output_subdir": OUT_SUBDIR_LOSSLESS_ADJUDICATION_V1,
        "mesh_file": "identical_to_L_mid_baseline_coupled_v2",
        "physics_model": "coupled_physical_core_v2",
        "seed_file": "diagnostics/acoustic_coupled_seed.npy",
        "seed_frequency_hz": SEED_F_HZ,
        "continuation_seed_applied": True,
        "actual_sigma_hz_target": APPROVED_SIGMA_HZ,
        "sigma_policy": "unregularized_offset_ladder_same_as_replacement",
        "st_type_persistence_required": True,
        "actual_st_a_shift_frac_target": 0.0,
        "actual_st_mass_reg_frac_target": 0.0,
        "eps_eigenvalue_semantics": "slepc_backtransformed",
        "legacy_double_shift_mapping_disabled": True,
        "diagnostic_operator_consistent_with_replay": True,
        "preserve_all_enabled": True,
        "eps_reject_sigma_spurious": False,
        "eps_reject_target_locked": False,
        "eps_reject_decoupled_u_only": False,
        "eps_harvest_allow_weak_coupling": True,
        "pre_replay_candidate_filtering_at_capture": False,
        "lossless_save_enabled": True,
        "lossless_replay_authoritative": True,
        "legacy_sparse_save": "comparison_only",
        "serializer_function_primary": "save_mode_dense_f64_lossless",
        "serializer_function_secondary": "save_mode_csr(dense_to_csr_f32_column)",
    }


def approved_replacement_policy_from_provenance(inventory: Dict[str, Any]) -> Dict[str, Any]:
    """Build approved-run policy dict from runtime provenance inventory."""
    out: Dict[str, Any] = {"source": "v2_mapping_fixed_replacement_runtime_provenance_inventory"}
    for row in inventory.get("fields") or []:
        if not isinstance(row, dict):
            continue
        name = row.get("field_name")
        if name:
            out[name] = row.get("selected_value_if_any")
    return out
