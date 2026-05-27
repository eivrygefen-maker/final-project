# ST singular-mass rehabilitation plan

Generated: 2026-05-27T09:33:37Z

## Next allowed action

- **next_allowed_action:** `review_null_basis_projection_preflight_no_eps`
- **VM command:** `None`

## PGNHEP / purification

- **Status:** `ruled_out_in_current_VM_environment`

## Stage 0 (implemented): eigenvalue mapping

- lam_phys=mu when eps_eigenvalue_semantics=slepc_backtransformed

## Current state (replacement baseline already executed)

- **Persistence self-test:** `PASS`
- **Replacement baseline EPS:** `completed_once`
- **Candidate persistence:** `56/56 closed`
- **Full pipeline audit:** `completed: MAPPING_FIXED_UNREGULARIZED_BASELINE_PERSISTED_VECTOR_CONTENT_UNRESOLVED`
- **Additional EPS solve:** `not authorized`
- **Report-only VM command:** `bash FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_mapping_fixed_persistence_fixed_full_pipeline_audit.sh`

## Historical: first mapping-corrected run

- **First-run persistence failure (inconclusive):** `False`
- **Not ST failure / not Stage-2:** True

## Stage 2

Not triggered by persistence failure. Mandatory only if replacement baseline persists and evaluates all candidates yet no physical branch is recovered.

